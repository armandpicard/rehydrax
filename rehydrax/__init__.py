from rehydrax._src.core import (
    rehydrate_stablehlo_module as rehydrate_stablehlo_module,
    rehydrate_stablehlo as rehydrate_stablehlo,
)

try:
    from rehydrax._src.torch2jax import rehydrate_torch_module as rehydrate_torch_module
except Exception:
    pass
