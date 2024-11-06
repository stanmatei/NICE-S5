from jax import lax
from jax.nn import initializers
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype
from flax.linen.module import Module, compact
from flax.linen.normalization import _canonicalize_axes, _abs_sq
from flax.typing import Array, PRNGKey, Dtype, Shape, Axes
from typing import Any, Callable, Optional, Tuple
import aqt.jax.v2.flax.aqt_flax as aqt
import jax
import jax.numpy as jnp
from .utils.quantization import q_dot_maybe, q_had_maybe
from .utils.shift_add_utils import round_power_of_2, round_to_fixed

def make_ste(f):
    @jax.custom_gradient
    def ste_foo(*args):
        return f(*args), lambda g: [g * arg_i for arg_i in args]
    
    return ste_foo

round_power_of_2_ste = make_ste(jax.tree_util.Partial(round_power_of_2, min = -14, max = 0))
round_to_fixed_ste = make_ste(jax.tree_util.Partial(round_to_fixed, fraction=16, integer=16))
    
class ShiftLinearLayer(nn.Module):
  weight: jax.Array = None
  hadamard: bool = False
  out_dim: int = 0
  use_bias: bool = False
  fraction_bits: int = 16
  integer_bits: int = 16
  use_complex: bool = False
  
  @nn.compact
  def __call__(self, x):

    if self.weight is not None:
      w = self.weight
    elif self.hadamard:
      w = self.param('w', nn.initializers.zeros, (x.shape[-1],))
    elif self.out_dim != 0:
      w = self.param('w', nn.initializers.lecun_normal(), (x.shape[-1], self.out_dim))
    else:
      w = self.param('w', nn.initializers.lecun_normal(), (x.shape[-1], x.shape[-1]))

    if self.use_bias:
      b = self.param('b', nn.initializers.zeros, (x.shape[-1]))
    else:
      b = None

    if self.use_complex:
        # TODO: consider coordinate-dependent transformations?
        w_imag_rounded = round_power_of_2_ste(jnp.imag(w))
        w_real_rounded = round_power_of_2_ste(jnp.real(w))
        w_rounded = w_real_rounded + 1j * w_imag_rounded

        x_imag_rounded = round_to_fixed_ste(jnp.imag(x))
        x_real_rounded = round_to_fixed_ste(jnp.real(x))
        x_rounded =  x_real_rounded + 1j * x_imag_rounded
    
    else:
        w_rounded = round_power_of_2_ste(w)
        x_rounded = round_to_fixed_ste(x)
    
    if self.hadamard:
      x = x_rounded * w_rounded
    else:
      x = jnp.matmul(x_rounded, w_rounded.T)

    if b is not None:
      b_rounded = round_to_fixed_ste(b)
      x = x + b_rounded

    #print("W shape", w_rounded.shape)

    return x

class DeltaLayer(nn.Module):
    thr: float = 0.0
    @nn.compact
    def __call__(self, x):
        x_roll = jnp.roll(x, 1, axis = 0)
        x_roll = x_roll.at[0, :].set(0.)
        delta = x - x_roll
        #delta = nn.relu(delta - thr) - nn.relu(-delta - thr)
        return delta

class PreAct(nn.Module):
    act: str = "relu"
    @nn.compact
    def __call__(self, x):
        if self.act == "relu":
            return nn.relu(x)
        elif self.act == "gelu":
            return nn.gelu(x)

class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    use_MLP_shift: bool = False
    use_sigma_delta: bool = False
    use_relu: bool = False
    thr: float = 0.0

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm()

        if self.use_relu:
            self.pre_act = PreAct()
        else:
            self.pre_act = PreAct(act = "gelu")

        if self.use_MLP_shift:
            if self.activation in ["full_glu"]:
                self.out1 = ShiftLinearLayer(use_bias = True)
                self.out2 = ShiftLinearLayer(use_bias = True)
            elif self.activation in ["half_glu1", "half_glu2"]:
                self.out2 = ShiftLinearLayer(use_bias = True)
        else:
            if self.activation in ["full_glu"]:
                self.out1 = nn.Dense(self.d_model)
                self.out2 = nn.Dense(self.d_model)
            elif self.activation in ["half_glu1", "half_glu2"]:
                self.out2 = nn.Dense(self.d_model)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.delta = DeltaLayer(thr=self.thr)

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)

        self.delta(x)

        if self.activation in ["full_glu"]:
            x = self.drop(self.pre_act(x))

            if self.use_sigma_delta:
                x = self.delta(x)
                out1 = jnp.cumsum(self.out1(x), axis = 0)
                out2 = jnp.cumsum(self.out2(x), axis = 0)
                x = out1 * jax.nn.sigmoid(out2)
            else:
                x = self.out1(x) * jax.nn.sigmoid(self.out2(x))

            x = self.drop(x)

        elif self.activation in ["half_glu1"]:
            x = self.drop(self.pre_act(x))

            if self.use_sigma_delta:
                x = self.delta(x)
                out2 = jnp.cumsum(self.out2(x), axis = 0)
                x = x * jax.nn.sigmoid(out2)
            else:
                x = x * jax.nn.sigmoid(self.out2(x))

            x = self.drop(x)

        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(self.pre_act(x))

            if self.use_sigma_delta:
                x1 = self.delta(x1)
                out2 = jnp.cumsum(self.out2(x1), axis = 0)
                x = x * jax.nn.sigmoid(out2)
            else:
                x = x * jax.nn.sigmoid(self.out2(x1))

            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(self.pre_act(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x
