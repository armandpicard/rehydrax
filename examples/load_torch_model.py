"""This example demonstrates how to convert a PyTorch model to a JAX model using rehydrax."""

import torch
import rehydrax
import jax
import jax.numpy as jnp


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear_1 = torch.nn.Linear(16, 32)
        self.linear_2 = torch.nn.Linear(32, 10)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear_2(x)
        return x


def main():
    model = MyModel()
    inputs_torch = (torch.randn((16,), dtype=torch.float32),)

    # Convert out model to jax
    model_state, model_f = rehydrax.rehydrate_torch_module(model, inputs_torch)
    inputs_jax = (jax.numpy.array(inputs_torch[0].detach().numpy()),)

    # Using our model as any other jax functions
    model_grad = jax.grad(lambda state, x: jnp.sum(model_f(state, x)))
    model_grad_jitted = jax.jit(model_grad)
    grad = model_grad_jitted(model_state, *inputs_jax)
    print(grad)


if __name__ == "__main__":
    main()
