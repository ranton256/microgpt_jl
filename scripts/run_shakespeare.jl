#!/usr/bin/env julia
"""
CLI entry point for MicroGPT Shakespeare training.
Trains a character-level GPT on tiny-shakespeare.txt with mini-batching.

Usage:
    julia scripts/run_shakespeare.jl
    julia scripts/run_shakespeare.jl --n_embd 64 --n_layer 2 --num_epochs 5
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using MicroGPT
using Printf: @printf

function parse_args()
    s = ArgParseSettings(description="MicroGPT Shakespeare: Character-level GPT on Shakespeare")
    @add_arg_table! s begin
        "--n_embd"
            arg_type = Int; default = 128
            help = "Embedding dimension"
        "--n_layer"
            arg_type = Int; default = 4
            help = "Number of transformer layers"
        "--block_size"
            arg_type = Int; default = 256
            help = "Maximum sequence length (context window)"
        "--n_head"
            arg_type = Int; default = 4
            help = "Number of attention heads"
        "--batch_size"
            arg_type = Int; default = 32
            help = "Number of chunks per training batch"
        "--num_epochs"
            arg_type = Int; default = 15
            help = "Number of passes over the dataset"
        "--learning_rate"
            arg_type = Float64; default = 3e-4
            help = "Peak learning rate"
        "--warmup_frac"
            arg_type = Float64; default = 0.1
            help = "Fraction of total steps for linear LR warmup"
        "--beta1"
            arg_type = Float64; default = 0.9
            help = "Adam beta1"
        "--beta2"
            arg_type = Float64; default = 0.98
            help = "Adam beta2"
        "--seed"
            arg_type = Int; default = 42
            help = "Random seed"
        "--num_samples"
            arg_type = Int; default = 3
            help = "Number of samples to generate after training"
        "--temperature"
            arg_type = Float64; default = 0.8
            help = "Sampling temperature for generation"
        "--sample_interval"
            arg_type = Int; default = 200
            help = "Generate samples every N steps (0 to disable)"
        "--log_interval"
            arg_type = Int; default = 10
            help = "Print loss every N steps"
        "--checkpoint"
            arg_type = String; default = "checkpoints/shakespeare.jld2"
            help = "Path to save the model checkpoint"
        "--no_save"
            action = :store_true
            help = "Skip saving checkpoint after training"
    end
    return ArgParse.parse_args(s)
end

function main()
    args = parse_args()
    device = get_device()

    # Load Shakespeare text
    text = load_shakespeare()
    @printf("text length: %d chars\n", length(text))
    flush(stdout)

    # Build tokenizer from the text
    tokenizer = CharTokenizer([text])
    @printf("vocab size: %d\n", tokenizer.vocab_size)

    # Chunk the text into fixed-length training sequences
    chunks = chunk_text(tokenizer, text, args["block_size"])
    @printf("num chunks: %d (block_size=%d)\n", length(chunks), args["block_size"])

    # Build batches
    batches = make_batches(chunks, args["batch_size"])
    @printf("num batches: %d (batch_size=%d)\n", length(batches), args["batch_size"])

    # Build model
    Random.seed!(args["seed"])
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=args["n_embd"],
        n_layer=args["n_layer"],
        block_size=args["block_size"],
        n_head=args["n_head"],
    )
    @printf("num params: %d\n", count_parameters(model))
    flush(stdout)

    # Train
    println("\n=== Training ===")
    flush(stdout)
    t_start = time()
    model, losses = train_batched!(model, batches;
        device=device,
        num_epochs=args["num_epochs"],
        learning_rate=args["learning_rate"],
        warmup_frac=args["warmup_frac"],
        beta1=args["beta1"],
        beta2=args["beta2"],
        seed=args["seed"],
        log_interval=args["log_interval"],
        sample_interval=args["sample_interval"],
        tokenizer=tokenizer,
    )
    t_elapsed = time() - t_start
    @printf("\nTraining complete in %.1fs (%.1f ms/step)\n",
            t_elapsed, t_elapsed / length(losses) * 1000)
    @printf("Final loss: %.4f (avg last 50: %.4f)\n",
            losses[end], sum(losses[max(1,end-49):end]) / min(50, length(losses)))

    # Save checkpoint
    if !args["no_save"]
        ckpt_path = args["checkpoint"]
        if !isabspath(ckpt_path)
            ckpt_path = joinpath(@__DIR__, "..", ckpt_path)
        end
        save_checkpoint(ckpt_path, model, tokenizer)
    end

    # Generate final samples
    println("\n=== Final Generation ===")
    generate_long(model, tokenizer;
        device=device,
        max_chars=args["block_size"],
        temperature=Float32(args["temperature"]),
        num_samples=args["num_samples"],
        seed=args["seed"],
    )
end

using Random
main()
