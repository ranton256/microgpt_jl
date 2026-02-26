#!/usr/bin/env julia
"""
CLI for generating Shakespeare text from a trained MicroGPT checkpoint.

Usage:
    julia scripts/generate_shakespeare.jl
    julia scripts/generate_shakespeare.jl --checkpoint checkpoints/shakespeare.jld2 --max_chars 500 --temperature 0.8 --num_samples 3
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using MicroGPT
using Printf: @printf

function parse_args()
    s = ArgParseSettings(description="Generate Shakespeare text from a trained MicroGPT checkpoint")
    @add_arg_table! s begin
        "--checkpoint"
            arg_type = String; default = "checkpoints/shakespeare.jld2"
            help = "Path to the model checkpoint (.jld2)"
        "--max_chars"
            arg_type = Int; default = 512
            help = "Maximum characters to generate per sample"
        "--temperature"
            arg_type = Float64; default = 0.8
            help = "Sampling temperature (lower = more deterministic)"
        "--num_samples"
            arg_type = Int; default = 3
            help = "Number of text samples to generate"
        "--seed"
            arg_type = Int; default = 42
            help = "Random seed"
    end
    return ArgParse.parse_args(s)
end

function main()
    args = parse_args()

    # Resolve checkpoint path relative to project root
    ckpt_path = args["checkpoint"]
    if !isabspath(ckpt_path)
        ckpt_path = joinpath(@__DIR__, "..", ckpt_path)
    end

    if !isfile(ckpt_path)
        error("Checkpoint not found: $ckpt_path\n" *
              "Train a model first with: julia scripts/run_shakespeare.jl")
    end

    # Load checkpoint
    @printf("Loading checkpoint: %s\n", ckpt_path)
    model, tokenizer = load_checkpoint(ckpt_path)
    @printf("Model: vocab_size=%d, n_embd=%d, n_layer=%d, block_size=%d\n",
            model.vocab_size, model.n_embd, length(model.blocks), model.block_size)
    @printf("Parameters: %d\n", count_parameters(model))

    # Device selection
    device = get_device()

    # Generate
    println("\n=== Generating Shakespeare ===\n")
    generate_long(model, tokenizer;
        device=device,
        max_chars=args["max_chars"],
        temperature=Float32(args["temperature"]),
        num_samples=args["num_samples"],
        seed=args["seed"],
    )
end

main()
