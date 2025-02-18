import torch
from typing import Callable
from torch import export
from torch_xla import stablehlo as torch_xla_stablehlo
from rehydrax._src import core
from jax._src.interpreters import mlir as mlir_interpreter
from jaxlib.mlir import ir
import numpy as np


def rehydrate_torch_module(
    module: torch.nn.Module, torch_sample_args
) -> tuple[dict, Callable]:
    """
    Rehydrates a PyTorch module into a callable function that can be used with JAX.

    This function takes a PyTorch module and sample arguments, and returns a state dictionary
    and a rehydrated function. The rehydrated function can be called with JAX arguments and
    transformed as a normal JAX functions

    Args:
        module (torch.nn.Module): The PyTorch module to rehydrate.
        torch_sample_args: Sample arguments to export the module.

    Returns:
        Tuple[Dict[str, np.ndarray], Callable]: A tuple containing the state dictionary and
        the rehydrated function.
    """

    def rehydrated_function(state, *jax_args):
        ctx = mlir_interpreter.make_ir_context()
        with ctx:
            exported = export.export(module, torch_sample_args)
            stablehlo_graph_module = torch_xla_stablehlo.exported_program_to_stablehlo(
                exported
            )
            forward = stablehlo_graph_module._name_to_stablehlo["forward"]
            args = []
            for loc in forward.meta.input_locations:
                if loc.position == -1:
                    args.append(state[loc.name])
                else:
                    args.append(jax_args[loc.position])

            hlo_module = ir.Module.parse(
                stablehlo_graph_module.get_stablehlo_text("forward")
            )

            return core.rehydrate_stablehlo_module(hlo_module)(*args)

    state = {k: np.asarray(v.detach().cpu()) for k, v in module.state_dict().items()}
    return state, rehydrated_function
