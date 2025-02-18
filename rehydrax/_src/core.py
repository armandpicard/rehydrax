from typing import Callable, Any

import jax

import numpy as np

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from typing import List
import jaxlib.mlir.dialects._stablehlo_ops_gen as stablehlo_ops
import jaxlib.mlir.dialects._chlo_ops_gen as chlo_ops

import jaxlib.mlir.dialects.stablehlo as stablehlo
from jax._src.interpreters import mlir as mlir_interpreter

import functools

import jax.numpy as jnp


def element_type_to_dtype(type_id: ir.TypeID) -> jnp.dtype:
    maping = {
        ir.F32Type.get(): jnp.float32,
        ir.IntegerType.get_signed(32): jnp.int32,
        ir.IntegerType.get_signless(32): jnp.int32,
        ir.IntegerType.get_signed(64): jnp.int64,
        ir.IntegerType.get_signless(64): jnp.int64,
        ir.IntegerType.get_unsigned(32): jnp.uint32,
        ir.IntegerType.get_unsigned(64): jnp.uint64,
        # TODO: other types
    }
    if type_id not in maping:
        raise RuntimeError(f"Unknown type_id {type_id}")
    return maping[type_id]


op_handler = {}


def register_handler(op_name: str, handler=None):
    if handler is None:
        return functools.partial(register_handler, op_name)
    op_handler[op_name] = handler
    return handler


def register_simple_handler(op_name: str, handler=None):
    if handler is None:
        return functools.partial(register_simple_handler, op_name)
    op_handler[op_name] = lambda m, o, *args: handler(o, *args)
    return handler


@register_simple_handler(stablehlo_ops.ConstantOp)
def handle_constant(op: stablehlo_ops.ConstantOp, *args):
    return (
        np.array(op.value, dtype=element_type_to_dtype(op.output.type.element_type)),
    )


@register_simple_handler(stablehlo_ops.CompareOp)
def handle_compare(op: stablehlo_ops.CompareOp, *args):
    assert len(args) == 2
    comparison_direction = stablehlo.ComparisonDirectionAttr(op.comparison_direction)
    mapping = {
        "EQ": jax.lax.eq,
        "NE": jax.lax.ne,
        "GE": jax.lax.ge,
        "GT": jax.lax.gt,
        "LE": jax.lax.le,
        "LT": jax.lax.lt,
    }
    # TODO: handle ComparisonType
    f = mapping[comparison_direction.value]
    return (f(*args),)


@register_simple_handler(stablehlo_ops.MaxOp)
def handle_max(op: stablehlo_ops.MaxOp, *args):
    return (jax.lax.max(*args),)


@register_handler(stablehlo_ops.ReduceOp)
def handle_reduce(module: ir.Module, op: stablehlo_ops.ReduceOp, *args):
    return jax.lax.reduce(
        args[: len(args) // 2],
        args[len(args) // 2 :],
        lambda a, b: rehydrate_stablehlo_block(module, op.body.blocks[0])(*a, *b),
        op.dimensions,
    )


@register_simple_handler(stablehlo_ops.BroadcastInDimOp)
def handle_broadcast_in_dim(op: stablehlo_ops.BroadcastInDimOp, *args):
    assert len(args) == 1

    return (
        jax.lax.broadcast_in_dim(
            args[0],
            op.result.type.shape,
            list(op.broadcast_dimensions),
        ),
    )


@register_simple_handler(stablehlo_ops.BroadcastOp)
def handle_broadcast(op: stablehlo_ops.BroadcastOp, *args):
    return (
        jax.lax.broadcast(
            args[0],
            list(op.broadcast_sizes),
        ),
    )


@register_simple_handler(stablehlo_ops.IotaOp)
def handle_iota(op: stablehlo_ops.IotaOp, *args):
    return (
        jax.lax.broadcasted_iota(
            element_type_to_dtype(op.result.type.element_type),
            op.result.type.shape,
            int(op.iota_dimension),
        ),
    )


@register_simple_handler(stablehlo_ops.ReshapeOp)
def handle_reshape(op: stablehlo_ops.ReshapeOp, *args):
    return (jax.numpy.reshape(args[0], list(op.result.type.shape)),)


@register_simple_handler(stablehlo_ops.ConvertOp)
def handle_convert(op: stablehlo_ops.ConvertOp, *args):
    assert len(args) == 1
    return (
        jax.lax.convert_element_type(
            args[0],
            element_type_to_dtype(op.result.type.element_type),
        ),
    )


@register_simple_handler(stablehlo_ops.ConcatenateOp)
def handle_concatenate(op: stablehlo_ops.ConcatenateOp, *args):
    return (
        jax.lax.concatenate(
            args,
            int(op.dimension),
        ),
    )


@register_simple_handler(stablehlo_ops.BitcastConvertOp)
def handle_bitcast_convert(op: stablehlo_ops.BitcastConvertOp, *args):
    assert len(args) == 1
    return (
        jax.lax.bitcast_convert_type(
            args[0],
            element_type_to_dtype(op.result.type.element_type),
        ),
    )


@register_simple_handler(stablehlo_ops.GatherOp)
def handle_gather(op: stablehlo_ops.GatherOp, *args):
    assert len(args) == 2

    dimension_numbers = stablehlo.GatherDimensionNumbers(op.dimension_numbers)

    return (
        jax.lax.gather(
            args[0],
            args[1],
            dimension_numbers=jax.lax.GatherDimensionNumbers(
                offset_dims=tuple(dimension_numbers.offset_dims),
                collapsed_slice_dims=tuple(dimension_numbers.collapsed_slice_dims),
                start_index_map=tuple(dimension_numbers.start_index_map),
                operand_batching_dims=tuple(dimension_numbers.operand_batching_dims),
            ),
            slice_sizes=op.slice_sizes,
            indices_are_sorted=bool(op.indices_are_sorted),
        ),
    )


@register_simple_handler(stablehlo_ops.SliceOp)
def handle_slice(op: stablehlo_ops.SliceOp, *args):
    assert len(args) == 1
    return (jax.lax.slice(*args, op.start_indices, op.limit_indices, op.strides),)


@register_simple_handler(stablehlo_ops.ReverseOp)
def handle_reverse(op: stablehlo_ops.ReverseOp, *args):
    assert len(args) == 1
    return (jax.lax.rev(args[0], list(op.dimensions)),)


@register_handler(stablehlo_ops.WhileOp)
def handle_while(module: ir.Module, op: stablehlo_ops.WhileOp, *args):
    def cond(args):
        block = op.cond.blocks[0]
        r = rehydrate_stablehlo_block(module, block)(*args)
        assert len(r) == 1
        return r[0]

    def body(args):
        block = op.body.blocks[0]
        r = rehydrate_stablehlo_block(module, block)(*args)
        return r

    return jax.lax.while_loop(
        cond,
        body,
        args,
    )


def _simple_function_handler(f, op: Any, *args):
    return (f(*args),)


simple_ops = {
    # StableHLO ops
    stablehlo_ops.AbsOp: jax.lax.abs,
    stablehlo_ops.AddOp: jax.lax.add,
    stablehlo_ops.AfterAllOp: jax.lax.after_all,
    stablehlo_ops.AndOp: jax.lax.bitwise_and,
    stablehlo_ops.Atan2Op: jax.lax.atan2,
    stablehlo_ops.CbrtOp: jax.lax.cbrt,
    stablehlo_ops.CeilOp: jax.lax.ceil,
    stablehlo_ops.ClampOp: jax.lax.clamp,
    stablehlo_ops.ClzOp: jax.lax.clz,
    stablehlo_ops.ComplexOp: jax.lax.complex,
    stablehlo_ops.CosineOp: jax.lax.cos,
    stablehlo_ops.CreateTokenOp: jax.lax.create_token,
    stablehlo_ops.DivOp: jax.lax.div,
    stablehlo_ops.ExpOp: jax.lax.exp,
    stablehlo_ops.Expm1Op: jax.lax.expm1,
    stablehlo_ops.FloorOp: jax.lax.floor,
    stablehlo_ops.ImagOp: jax.lax.imag,
    stablehlo_ops.IsFiniteOp: jax.lax.is_finite,
    stablehlo_ops.LogOp: jax.lax.log,
    stablehlo_ops.Log1pOp: jax.lax.log1p,
    stablehlo_ops.LogisticOp: jax.lax.logistic,
    stablehlo_ops.MaxOp: jax.lax.max,
    stablehlo_ops.MinOp: jax.lax.min,
    stablehlo_ops.MulOp: jax.lax.mul,
    stablehlo_ops.NegOp: jax.lax.neg,
    stablehlo_ops.NotOp: jax.lax.bitwise_not,
    stablehlo_ops.OrOp: jax.lax.bitwise_or,
    stablehlo_ops.OptimizationBarrierOp: jax.lax.optimization_barrier,
    stablehlo_ops.PopulationCountOp: jax.lax.population_count,
    stablehlo_ops.PowOp: jax.lax.pow,
    stablehlo_ops.RealOp: jax.lax.real,
    stablehlo_ops.RemOp: jax.lax.rem,
    stablehlo_ops.RoundOp: jax.lax.round,
    stablehlo_ops.RsqrtOp: jax.lax.rsqrt,
    stablehlo_ops.SelectOp: jax.lax.select,
    stablehlo_ops.ShiftLeftOp: jax.lax.shift_left,
    stablehlo_ops.ShiftRightArithmeticOp: jax.lax.shift_right_arithmetic,
    stablehlo_ops.ShiftRightLogicalOp: jax.lax.shift_right_logical,
    stablehlo_ops.SignOp: jax.lax.sign,
    stablehlo_ops.SineOp: jax.lax.sin,
    stablehlo_ops.SqrtOp: jax.lax.sqrt,
    stablehlo_ops.SubtractOp: jax.lax.sub,
    stablehlo_ops.TanOp: jax.lax.tan,
    stablehlo_ops.TanhOp: jax.lax.tanh,
    stablehlo_ops.XorOp: jax.lax.bitwise_xor,
    chlo_ops.ErfInvOp: jax.lax.erf_inv,
    chlo_ops.ErfOp: jax.lax.erf,
    chlo_ops.NextAfterOp: jax.lax.nextafter,
}

for op, f in simple_ops.items():
    register_simple_handler(op, functools.partial(_simple_function_handler, f))


@register_simple_handler(stablehlo_ops.TransposeOp)
def handle_transpose(op: stablehlo_ops.TransposeOp, *args):
    assert len(args) == 1
    return (jax.lax.transpose(args[0], list(op.permutation)),)


@register_simple_handler(stablehlo_ops.DotOp)
def handle_dot(op: stablehlo_ops.DotOp, *args):
    assert len(args) == 2
    # TODO: precision_config
    return (jax.lax.dot(args[0], args[1]),)


@register_simple_handler(stablehlo_ops.DotGeneralOp)
def handle_dot_general(op: stablehlo_ops.DotGeneralOp, *args):
    assert len(args) == 2
    dot_dimension_numbers = stablehlo.DotDimensionNumbers(op.dot_dimension_numbers)
    # TODO(armand): handle precision and others

    return (
        jax.lax.dot_general(
            args[0],
            args[1],
            dimension_numbers=(
                (
                    list(dot_dimension_numbers.lhs_contracting_dimensions),
                    list(dot_dimension_numbers.rhs_contracting_dimensions),
                ),
                (
                    list(dot_dimension_numbers.lhs_batching_dimensions),
                    list(dot_dimension_numbers.rhs_batching_dimensions),
                ),
            ),
            # TODO: precision_config, algorithm
        ),
    )


@register_simple_handler(stablehlo_ops.ConvolutionOp)
def handle_convolution(op: stablehlo_ops.ConvolutionOp, *args):
    assert len(args) == 2

    padding = []
    for i in range(len(op.window_strides)):
        padding.append((op.padding[i * 2], op.padding[i * 2 + 1]))

    dimension_numbers = stablehlo.ConvDimensionNumbers(op.dimension_numbers)

    return (
        jax.lax.conv_general_dilated(
            args[0],
            args[1],
            window_strides=tuple(op.window_strides),
            padding=tuple(padding),
            lhs_dilation=tuple(op.lhs_dilation),
            rhs_dilation=tuple(op.rhs_dilation),
            dimension_numbers=jax.lax.ConvDimensionNumbers(
                lhs_spec=tuple(
                    [
                        int(dimension_numbers.input_batch_dimension),
                        int(dimension_numbers.input_feature_dimension),
                        *list(dimension_numbers.input_spatial_dimensions),
                    ]
                ),
                rhs_spec=tuple(
                    [
                        int(dimension_numbers.kernel_output_feature_dimension),
                        int(dimension_numbers.kernel_input_feature_dimension),
                        *list(dimension_numbers.kernel_spatial_dimensions),
                    ]
                ),
                out_spec=tuple(
                    [
                        int(dimension_numbers.output_batch_dimension),
                        int(dimension_numbers.output_feature_dimension),
                        *list(dimension_numbers.output_spatial_dimensions),
                    ]
                ),
            ),
            feature_group_count=int(op.feature_group_count),
            batch_group_count=int(op.batch_group_count),
            # TODO(armand): precision and others
        ),
    )


@register_handler(stablehlo_ops.ReducePrecisionOp)
def handle_reduce_precision(op: stablehlo_ops.ReducePrecisionOp, *args):
    return (jax.lax.reduce_precision(args[0], op.exponent_bits, op.mantissa_bits),)


@register_handler(func_dialect.CallOp)
def handle_call(module: ir.Module, op: func_dialect.CallOp, *args):
    callee_block = None

    for block in module.body.operations:
        if block.name.value == op.callee.value:
            callee_block = block
    assert callee_block is not None
    return rehydrate_stablehlo_block(module, callee_block.body.blocks[0])(*args)


# TODO: Ops not yes handled. Some maybe not be possible.
# AllGatherOp
# AllReduceOp
# AllToAllOp
# BatchNormGradOp
# BatchNormInferenceOp
# BatchNormTrainingOp
# CaseOp
# CholeskyOp
# CollectiveBroadcastOp
# CollectivePermuteOp
# CompositeOp
# CrossReplicaSumOp
# CustomCallOp
# DynamicBroadcastInDimOp
# DynamicConvOp
# DynamicGatherOp
# DynamicIotaOp
# DynamicPadOp
# DynamicReshapeOp
# DynamicSliceOp
# DynamicUpdateSliceOp
# EinsumOp
# FftOp
# GetDimensionSizeOp
# GetTupleElementOp
# IfOp
# MapOp
# OutfeedOp
# PadOp
# PartitionIdOp
# RealDynamicSliceOp
# RecvOp
# ReduceScatterOp
# ReduceWindowOp
# ReplicaIdOp
# RngBitGeneratorOp
# RngOp
# RoundNearestEvenOp
# ScatterOp
# SelectAndScatterOp
# SendOp
# SetDimensionSizeOp
# SortOp
# TorchIndexSelectOp
# TriangularSolveOp
# TupleOp
# UnaryEinsumOp
# UniformDequantizeOp
# UniformQuantizeOp


def check_arguments(a: ir.BlockArgumentList, b: List[jax.Array]):
    for arg, value in zip(a, b):
        assert list(arg.type.shape) == list(value.shape), (
            f"{arg.type.shape=} {value.shape=}"
        )


def rehydrate_stablehlo_block(module: ir.Module, block: ir.Block) -> Callable:
    def rehydrated_function(*args: list[jax.Array]):
        env = {}

        def read(var):
            return env[var.get_name()]

        def write(var, val):
            env[var.get_name()] = val

        assert len(block.arguments) == len(args)
        check_arguments(block.arguments, args)
        for arg, value in zip(block.arguments, args):
            write(arg, value)

        for j, op in enumerate(block):
            if type(op) in op_handler:
                handler = op_handler[type(op)]
                args = [read(var) for var in op.operands]

                results = handler(module, op, *args)
                if len(op.results) != len(results):
                    assert len(op.results) == len(results)

                for var, val in zip(op.results, results):
                    write(var, val)

            elif isinstance(op, stablehlo_ops.ReturnOp) or isinstance(
                op, func_dialect.ReturnOp
            ):
                return tuple(read(var) for var in op.operands)

            else:
                raise NotImplementedError(f"No handler for {type(op)}")
        return None

    return rehydrated_function


def rehydrate_stablehlo_module(module: ir.Module) -> Callable:
    def rehydrated_stablehlo_module(*args):
        res = rehydrate_stablehlo_block(
            module, module.body.operations[0].body.blocks[0]
        )(*args)
        if len(res) == 1:
            return res[0]
        return res

    return rehydrated_stablehlo_module


def rehydrate_stablehlo(stablehlo_text: str) -> Callable:
    def rehydrated_function(*args: list[jax.Array]):
        ctx = mlir_interpreter.make_ir_context()
        with ctx:
            module = ir.Module.parse(stablehlo_text)
            return rehydrate_stablehlo_module(module)(*args)

    return rehydrated_function
