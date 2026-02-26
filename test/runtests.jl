using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using Flux
using Flux: cpu, softmax
using Statistics: mean
using Random
using Printf: @printf
using MicroGPT

# Helper function to capture stdout (Julia 1.12 compatible)
function capture_out(f)
    rd, wr = redirect_stdout()
    try
        f()
        flush(stdout)
    finally
        redirect_stdout()
        close(wr)
    end
    output = read(rd, String)
    close(rd)
    return output
end

@testset "MicroGPT" begin
    include("test_tokenizer.jl")
    include("test_model.jl")
    include("test_train.jl")
    include("test_checkpoint.jl")
    include("test_integration.jl")
end
