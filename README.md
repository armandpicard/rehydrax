# Rehydrax
Rehydrate stablehlo into jax.

> [!WARNING]
> This code is still experimental, all ops are not handled yet.

## Install
### Jax only
```
pip install rehydrax
```
### with torch
```
pip install rehydrax[torch] torch torch-xla-cuda-plugin@https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla_cuda_plugin-2.5.0-py3-none-any.whl
```

## Examples

### Loading pytorch model into jax 

```python
model = MyModel()
inputs_torch = (torch.randn((16,), dtype=torch.float32),)

# Convert out model to jax
model_state, model_f = rehydrax.rehydrate_torch_module(model, inputs_torch)
inputs_jax = (jax.numpy.array(inputs_torch[0].detach().numpy()),)

# Using our model as any other jax functions
model_grad = jax.grad(lambda state, x: jnp.sum(model_f(state, x)))
model_grad_jitted = jax.jit(model_grad)
grad = model_grad_jitted(model_state, *inputs_jax)
```

### Loading model from other project
You may wan't to just try to use model from another project but don't wan't to depend on that project for any reason. You can do that by just exporting the model from project A and just rehydrate the model on project B.

#### Project A
```python

@jax.jit
def init(rng):
    model = Model(rngs=nnx.Rngs(rng))
    return nnx.split(model)[1]

@jax.jit
def forward(state, x):
    model = Model(nnx.Rngs(0))
    model_graph = nnx.split(model)[0]
    model = nnx.merge(model_graph, state)
    return model(x)

key = jax.random.PRNGKey(0)

init_lowered = init.lower(key)
init_stablehlo = init_lowered.as_text()
state_abstract = jax.eval_shape(init, key)

sample = jax.random.uniform(key, (1, 2), jnp.float32)
forward_lowered = forward.lower(state_abstract, sample)
forward_stablehlo = forward_lowered.as_text()
```

#### Project B
```python
init_rehydrated = rehydrax.rehydrate_stablehlo(init_stablehlo)
state = init_rehydrated(key)
forward_rehydrated = rehydrax.rehydrate_stablehlo(forward_stablehlo)
y2 = forward_rehydrated(*state, sample)
```
