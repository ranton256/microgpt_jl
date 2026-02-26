module MicroGPT

using Flux
using Flux: Embedding, Dense, Adam, softmax
using NNlib
using Statistics: mean
using Random
using Random: shuffle
using StatsBase: Weights, sample
using Printf: @printf

# ---------------------------------------------------------------------------- #
# Workaround: NNlib's batched_gemm! only has a BLAS method for CPU Arrays.
# GPU backends like Metal lack a batched GEMM kernel, so NNlib.batched_mul
# throws a MethodError on MtlArrays (NNlib issue #581).
#
# We add a generic fallback that uses per-slice matrix multiplication,
# which works on any backend that supports standard `*` (Metal, etc.).
# The existing Array-specific method remains preferred for CPU arrays.
# ---------------------------------------------------------------------------- #
function NNlib._batched_gemm!(::Type, transA::Char, transB::Char,
                               α::Number, A, B, β::Number, C)
    for k in axes(C, 3)
        Ak = transA == 'T' ? permutedims(@view(A[:, :, k])) : @view(A[:, :, k])
        Bk = transB == 'T' ? permutedims(@view(B[:, :, k])) : @view(B[:, :, k])
        if β == zero(β)
            C[:, :, k] .= α .* (Ak * Bk)
        else
            C[:, :, k] .= α .* (Ak * Bk) .+ β .* C[:, :, k]
        end
    end
end

"""
Detect the best available compute device.
Returns a function that moves arrays/models to the device.

Note: GPU backends (Metal.jl, CUDA.jl) must be loaded by the caller
before calling this function. For example:
    using Metal   # or: using CUDA
    using MicroGPT
    device = get_device()   # → returns gpu
"""
function get_device()
    dev = Flux.get_device()
    devname = string(typeof(dev))
    if contains(devname, "CUDA")
        println("Using CUDA GPU")
        return gpu
    elseif contains(devname, "Metal")
        println("Using Metal GPU")
        return gpu
    else
        println("Using CPU")
        return cpu
    end
end

include("tokenizer.jl")
include("model.jl")
include("train.jl")
include("checkpoint.jl")

export CharTokenizer, encode, decode, bos_id
export GPT, RMSNorm, count_parameters, KVCache, generate_step
export load_dataset, train!, generate, get_device
export load_shakespeare, chunk_text, make_batches, train_batched!, generate_long
export save_checkpoint, load_checkpoint

end # module
