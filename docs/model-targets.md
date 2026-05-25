# Model Targets

Wan2.2 and LTX2.3 are the initial target model families.

The implementation should use backend adapter interfaces so exact model loading logic can be replaced later. Model-specific setup belongs in adapters, not in the profiler core.

## Initial Targets

### Wan2.2

The Wan2.2 adapter should expose the common profiler phases while allowing the underlying MLX model loading and pipeline implementation to change.

### LTX2.3

The LTX2.3 adapter should follow the same adapter contract as Wan2.2 so benchmark records remain comparable across model families.

## Core Constraint

Do not hard-code one model pipeline into the profiler core. The profiler coordinates timing, synchronization, metrics, and reporting through common interfaces.
