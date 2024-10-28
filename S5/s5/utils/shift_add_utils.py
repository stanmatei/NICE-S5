import jax.numpy as jnp
import flax
import jax

def round_to_fixed(input, fraction=16, integer=16):
    assert integer >= 1, integer
    if integer == 1:
        return jnp.sign(input) - 1
    delta = jnp.pow(2.0, -(fraction))
    bound = jnp.pow(2.0, integer-1)
    min_val = - bound
    max_val = bound - 1
    rounded = jnp.floor(input / delta) * delta
    clipped_value = jnp.clip(rounded, min_val, max_val)
    return clipped_value

def get_shift_and_sign(x, rounding='deterministic'):
    sign = jnp.sign(x)

    x_abs = jnp.abs(x)
    shift = round(jnp.log(x_abs) / jnp.log(2), rounding)

    return shift, sign

def get_shift_and_sign(x, rounding='deterministic'):
    sign = jnp.sign(x)

    x_abs = jnp.abs(x)
    shift = round(jnp.log(x_abs) / jnp.log(2), rounding)

    return shift, sign

def round_power_of_2(x, rounding='deterministic'):
    shift, sign = get_shift_and_sign(x, rounding)
    # print(shift)
    x_rounded = (2.0 ** shift) * sign
    return x_rounded


def round(x, rounding='deterministic'):
    assert(rounding in ['deterministic', 'stochastic'])
    if rounding == 'stochastic':
        x_floor = jnp.floor(x)
        key = jax.random.key(42)
        return x_floor + jax.random.bernoulli(key, x - x_floor)
    else:
        return jnp.round(x)