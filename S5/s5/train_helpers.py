from functools import partial
import jax
import jax.numpy as np
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
from flax.core.frozen_dict import unfreeze
import optax
from typing import Any, Tuple


def _compute_act_sparsity(act, atol=1e-6):
    return np.isclose(act, 0, atol=atol).sum() / act.size


# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step,  lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = np.array(lr_val, dtype=np.float32)
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)
    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       in_dim=1,
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       grad_clip_threshold=None,
                       dt_global=False
                       ):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param grad_clip_threshold:
    :param dt_global:
    :return:
    """

    def inject_hyperparams_clipped(opt_cls):
        def inner_func(**kwargs):
            if grad_clip_threshold is None:
                return optax.inject_hyperparams(opt_cls)(**kwargs)
            else:
                return optax.chain(
                    optax.inject_hyperparams(opt_cls)(**kwargs),
                    optax.clip_by_global_norm(grad_clip_threshold)
                )
        return inner_func

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*bsz, seq_len, in_dim)), np.ones(2*bsz))
            integration_timesteps = np.ones((2*bsz, seq_len,))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones((bsz, seq_len,))
    else:
        dummy_input = np.ones((bsz, seq_len, in_dim))
        integration_timesteps = np.ones((bsz, seq_len, ))

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           dummy_input, integration_timesteps,
                           )
    if batchnorm:
        # params = variables["params"].unfreeze()
        params = unfreeze(variables["params"])
        batch_stats = variables["batch_stats"]
    else:
        # params = variables["params"].unfreeze()
        params = unfreeze(variables["params"])
        # Note: `unfreeze()` is for using Optax.

    if opt_config in ["qaft"]:
        print("using quantization-aware fine-tuning optimization setup (SGD, no decay)")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.sgd)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.sgd)(learning_rate=ssm_lr, momentum=0.9),
                "regular": inject_hyperparams_clipped(optax.sgd)(learning_rate=lr, momentum=0.9),
            },
            ssm_fn,
        )
    elif opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.sgd)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.adam)(learning_rate=ssm_lr),
                "regular": inject_hyperparams_clipped(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": inject_hyperparams_clipped(optax.adam)(learning_rate=ssm_lr),
                "regular": inject_hyperparams_clipped(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply 
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.adamw)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.adam)(learning_rate=ssm_lr),
                "regular": inject_hyperparams_clipped(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.sgd)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.adam)(learning_rate=ssm_lr),
                "regular": inject_hyperparams_clipped(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Train and eval steps
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


def prep_batch(batch: tuple,
               seq_len: int,
               in_dim: int) -> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # Convert to JAX.
    if not isinstance(inputs, np.ndarray):
        inputs = np.asarray(inputs.numpy())

    # Grab lengths from aux if it is there.
    lengths = aux_data.get('lengths', None)

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = np.pad(inputs, ((0, 0), (0, num_pad)), 'constant', constant_values=(0,))

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(np.asarray(inputs), in_dim)

    # If there are lengths, bundle them up.
    if lengths is not None:
        lengths = np.asarray(lengths.numpy())
        full_inputs = (inputs.astype(float), lengths.astype(float))
    else:
        full_inputs = inputs.astype(float)

    # Convert and apply.
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets.numpy())

    # If there is an aux channel containing the integration times, then add that.
    if 'timesteps' in aux_data.keys():
        integration_timesteps = np.diff(np.asarray(aux_data['timesteps'].numpy()))
    else:
        integration_timesteps = np.ones((len(inputs), seq_len))

    return full_inputs, targets.astype(float), integration_timesteps


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, batchnorm, lr_params, loss_fn=cross_entropy_loss, log_act_sparsity=False):
    """
    Training function for an epoch that loops over batches.
    """
    # Store Metrics
    model = model(training=True)
    batch_losses = []
    act_sparsities = []

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        # rng, drop_rng = jax.random.split(rng)
        state, loss, act_sp = train_step(
            state,
            rng,  # move splitting into train_step
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
            loss_act=loss_fn,
            log_act_sparsity=log_act_sparsity,
        )
        batch_losses.append(loss)
        if log_act_sparsity:
            act_sparsities.append(act_sp)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

    if log_act_sparsity:
        # aggregate mean
        act_sparsities = jax.tree.map(lambda *xs: np.mean(np.stack(xs)), *act_sparsities)
        act_sparsities = jax.tree.map(lambda x: x if x is None else x.item(), act_sparsities)
        act_sparsities = jax.tree.map(lambda x: x[0] if isinstance(x, tuple) else x, act_sparsities)
        # clean up
        clean_up_fn = lambda x: x['__call__'] if isinstance(x, dict) and '__call__' in x else x
        is_leaf_fn = lambda node: isinstance(node, dict) and '__call__' in node
        act_sparsities = jax.tree.map(clean_up_fn, act_sparsities, is_leaf=is_leaf_fn)
        # flatten
        act_sparsities = flatten_pytree(act_sparsities)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step, act_sparsities


def validate(state, skey, model, testloader, seq_len, in_dim, batchnorm, 
             loss_fn=cross_entropy_loss, calculate_acc=True, step_rescale=1.0):
    """Validation function that loops over batches"""
    model = model(training=False, step_rescale=step_rescale)
    losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, in_dim)
        loss, acc, pred = eval_step(inputs, labels, skey, integration_timesteps, state, model, batchnorm, loss_act=loss_fn, calculate_acc=calculate_acc)
        losses = np.append(losses, loss)
        if calculate_acc:
            accuracies = np.append(accuracies, acc)

    aveloss = np.mean(losses)
    aveaccu = None
    if calculate_acc:
        aveaccu = np.mean(accuracies)
    return aveloss, aveaccu


@partial(jax.jit, static_argnums=(5, 6, 7))
def train_step(state,
               rng,
               batch_inputs,
               batch_labels,
               batch_integration_timesteps,
               model,
               batchnorm,
               loss_act = cross_entropy_loss,
               log_act_sparsity = False,
               ):
    """Performs a single training step given a batch of data"""
    rng, drop_rng = jax.random.split(rng)  # moved here from train_epoch
    def loss_fn(params):

        if batchnorm:
            # adding params rng to make aqt/flax happy
            logits, mod_vars = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": drop_rng, "params": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": drop_rng, "params": rng},
                mutable=["intermediates"],
            )

        loss = np.mean(loss_act(logits, batch_labels))

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if log_act_sparsity:
        act_sparsity_logs = jax.tree_util.tree_map(lambda x: _compute_act_sparsity(x), mod_vars["intermediates"])
    else:
        act_sparsity_logs = None

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss, act_sparsity_logs


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def eval_step(batch_inputs,
              batch_labels,
              skey,
              batch_integration_timesteps,
              state,
              model,
              batchnorm,
              loss_act=cross_entropy_loss,
              calculate_acc=False
              ):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             batch_inputs, batch_integration_timesteps,
                             rngs={"params": skey},
                             )
    else:
        logits = model.apply({"params": state.params},
                             batch_inputs, batch_integration_timesteps,
                             rngs={"params": skey},
                             )

    losses = loss_act(logits, batch_labels)
    accs = None
    if calculate_acc:
        accs = compute_accuracy(logits, batch_labels)

    return losses, accs, logits
