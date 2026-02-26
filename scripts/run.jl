#!/usr/bin/env julia
"""
CLI entry point for MicroGPT training and generation.
Mirrors the Python argparse interface.

Usage:
    julia scripts/run.jl --n_embd 16 --n_layer 1 --block_size 8 --num_steps 1000 --n_head 4 --learning_rate 0.01 --seed 42
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Load GPU backend (if available) before MicroGPT
try using CUDA; using cuDNN catch end
try using Metal catch end

using ArgParse
using MicroGPT

function parse_args()
    s = ArgParseSettings(description="MicroGPT: A minimal GPT trainer in Julia")
    @add_arg_table! s begin
        "--n_embd"
            arg_type = Int
            default = 16
            help = "Embedding dimension"
        "--n_layer"
            arg_type = Int
            default = 1
            help = "Number of transformer layers"
        "--block_size"
            arg_type = Int
            default = 16
            help = "Maximum sequence length"
        "--n_head"
            arg_type = Int
            default = 4
            help = "Number of attention heads"
        "--num_steps"
            arg_type = Int
            default = 1000
            help = "Number of training steps"
        "--learning_rate"
            arg_type = Float64
            default = 0.01
            help = "Learning rate"
        "--seed"
            arg_type = Int
            default = 42
            help = "Random seed"
        "--num_samples"
            arg_type = Int
            default = 5
            help = "Number of samples to generate after training"
        "--temperature"
            arg_type = Float64
            default = 0.5
            help = "Sampling temperature for generation"
    end
    return ArgParse.parse_args(s)
end

function main()
    args = parse_args()

    device = get_device()

    # Load dataset
    docs = load_dataset()
    println("num docs: $(length(docs))")

    # Build tokenizer
    tokenizer = CharTokenizer(docs)
    println("vocab size: $(tokenizer.vocab_size)")

    # Build model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=args["n_embd"],
        n_layer=args["n_layer"],
        block_size=args["block_size"],
        n_head=args["n_head"],
    )
    println("num params: $(count_parameters(model))")

    # Train
    model = train!(model, tokenizer, docs;
        device=device,
        num_steps=args["num_steps"],
        learning_rate=args["learning_rate"],
        seed=args["seed"],
    )

    # Generate
    generate(model, tokenizer;
        device=device,
        num_samples=args["num_samples"],
        temperature=Float32(args["temperature"]),
        seed=args["seed"],
    )
end

main()
