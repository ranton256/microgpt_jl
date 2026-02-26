module MicroGPT

using Flux
using Flux: Embedding, Dense, Adam, softmax
using NNlib
using Statistics: mean
using Random
using Random: shuffle
using StatsBase: Weights, sample
using Printf: @printf

# Optional GPU backends
try using CUDA catch end
try using Metal catch end

"""
Detect the best available compute device.
Returns a function that moves arrays/models to the device.
"""
function get_device()
    if @isdefined(CUDA) && CUDA.functional()
        println("Using CUDA GPU")
        return gpu
    elseif @isdefined(Metal) && Metal.functional()
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
