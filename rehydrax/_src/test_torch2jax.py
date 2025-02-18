import torch
import numpy as np
import pytest
import jax

from rehydrax import rehydrate_torch_module

from typing import Any


@pytest.fixture
def sample_input_numpy(sample_input_torch: Any):
    return jax.tree.map(np.array, sample_input_torch)


@pytest.mark.parametrize(
    "module, sample_input_torch",
    [
        (
            torch.nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            (
                torch.tensor(np.random.uniform(size=(1, 256, 16))).to_dense(
                    torch.float32
                ),
            ),
        ),
        (
            torch.nn.Linear(16, 2),
            (torch.tensor(np.random.uniform(size=(1, 16))).to_dense(torch.float32),),
        ),
        (
            torch.nn.CrossEntropyLoss(),
            (
                torch.tensor(np.random.uniform(size=(1, 16))).to_dense(torch.float32),
                torch.tensor(np.random.randint(0, 2, size=(1,))).to_dense(torch.long),
            ),
        ),
    ],
)
def test_module(module, sample_input_torch, sample_input_numpy):
    y_torch = module(*sample_input_torch)
    state, rehydrated_function = rehydrate_torch_module(module, sample_input_torch)
    y_jax = rehydrated_function(state, *sample_input_numpy)
    assert np.allclose(y_jax, y_torch.detach().numpy())
