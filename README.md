# MicroGPT.jl

A Julia port of Andrej Karpathy's [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) — a minimal, educational implementation of a GPT language model. The original is 200 lines of pure Python with zero dependencies. This port brings the same ideas to Julia using [Flux.jl](https://fluxml.ai/), then extends them to train on Shakespeare.

## Background

Karpathy's microgpt distills a decade of work (micrograd, makemore, nanoGPT) into the simplest possible GPT: tokenization, autograd, multi-head attention, Adam optimizer, training, and inference — all in a single file. As he puts it, everything beyond those ~200 lines is "just efficiency."

This Julia port started as a faithful 1:1 reimplementation. We first matched the Python original exactly — same architecture (RMSNorm, ReLU, no biases, separate lm_head), same hyperparameters, same 4,192-parameter model, same names dataset. Once the two implementations produced equivalent results, we scaled up to a much more interesting task: training on Shakespeare's collected works.

## Motivation

Karpathy's original microgpt is pure Python with no dependencies — beautiful for understanding the algorithms, but slow to train anything beyond a toy dataset. You could reach for PyTorch to speed things up, but reimplementing in Julia felt like a more natural fit.

Julia was designed in the age of GPUs, and it shows. GPU acceleration isn't bolted on after the fact — it's woven into the language and ecosystem from the ground up. With [Flux.jl](https://fluxml.ai/), you write your model once and `Flux.gpu` sends it to whatever accelerator is available: CPU, NVIDIA GPUs via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), or Apple GPUs via [Metal.jl](https://github.com/JuliaGPU/Metal.jl) — no code changes required. Julia's multiple dispatch and JIT compilation also mean the CPU path is fast out of the box (our names training runs 70x faster than the pure-Python original), making it a practical choice for taking microgpt from a pedagogical exercise to something you can train on real text in a reasonable amount of time.

## How it was built

This project was built collaboratively with [Claude Code](https://claude.ai/claude-code) (Anthropic's AI coding agent). The development followed a natural progression:

1. **Initial port** — Translate the Python microgpt to idiomatic Julia with Flux.jl, getting the names-generation workflow running.
2. **Architecture alignment** — Systematically match all 7 architectural differences against the Python original (BOS/EOS handling, weight tying, activation function, init scale, Adam betas, block size, output projection init) until the Julia model produced numerically equivalent results with identical parameter counts.
3. **Shakespeare scaling** — Add mini-batched epoch training with cosine LR scheduling and linear warmup, chunked text encoding, and continuous text generation to handle the 1.1M-character tiny-shakespeare dataset.
4. **Checkpoint persistence** — Add JLD2-based model serialization so you can train once and generate forever, plus a standalone inference CLI.
5. **Test suite** — 157 test assertions across 5 files covering tokenizer, model, training, checkpoints, and end-to-end integration.

Each step was planned, implemented, tested, and committed before moving to the next — the git history tells the full story.

## Features

- **Character-level GPT** matching Karpathy's architecture: RMSNorm, ReLU, no biases, separate lm_head
- **Two workflows**: names generation (toy, trains in seconds) and Shakespeare (trains in ~15 minutes on CPU)
- **Mini-batched training** with linear warmup + cosine LR decay
- **KV-cached autoregressive generation** for efficient inference
- **Checkpoint save/load** via JLD2 — train once, generate anytime
- **CLI scripts** with full ArgParse support for all hyperparameters
- **Comprehensive test suite** (157 assertions, runs in ~25 seconds)

## Quick Start

### Prerequisites

- Julia 1.10+ (tested on 1.12)

### Setup

```bash
git clone https://github.com/yourusername/microgpt_jl.git
cd microgpt_jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Train on Names (the original microgpt task)

```bash
julia scripts/run.jl
```

This downloads the [names dataset](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt) (32K names, 27-char vocab), trains a tiny 4,192-parameter model in about 1 second, and generates new names:

```
num docs: 32033
vocab size: 27
num params: 4192
step    1 / 1000 | loss 3.6063
step    2 / 1000 | loss 3.3504
...
step 1000 / 1000 | loss 2.2959

--- inference (new, hallucinated names) ---
sample  1: karin
sample  2: kenan
sample  3: eleele
sample  4: karen
sample  5: darin
```

### Train on Shakespeare

```bash
julia scripts/run_shakespeare.jl
```

This downloads [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1.1M chars, 65-char vocab), trains a larger model, and auto-saves a checkpoint:

```
text length: 1115394 chars
vocab size: 66
num chunks: 4341 (block_size=256)
num batches: 135 (batch_size=32)
num params: 357536

=== Training ===
total steps: 2025 (135 batches × 15 epochs), warmup: 202 steps
epoch  1 | step     1 / 2025 | lr 1.49e-06 | loss 4.2015
epoch  1 | step    10 / 2025 | lr 1.49e-05 | loss 4.0757
...
epoch 15 | step  2025 / 2025 | lr 1.03e-08 | loss 1.5832

Training complete in 807.0s (398.5 ms/step)
Final loss: 1.5832 (avg last 50: 1.7456)
Checkpoint saved to checkpoints/shakespeare.jld2
```

The default configuration uses `n_embd=128, n_layer=4, n_head=4, block_size=256` for a 357K-parameter model. You can experiment with smaller models for faster iteration:

```bash
# Quick smoke test (~1 minute)
julia scripts/run_shakespeare.jl --n_embd 32 --n_layer 1 --num_epochs 3 --batch_size 8

# Larger model
julia scripts/run_shakespeare.jl --n_embd 192 --n_layer 6 --n_head 6 --num_epochs 20
```

### Generate Shakespeare (from saved checkpoint)

Once you've trained a model, generate text anytime without retraining:

```bash
julia scripts/generate_shakespeare.jl
```

Customize generation:

```bash
julia scripts/generate_shakespeare.jl --temperature 1.0 --num_samples 5 --max_chars 256
julia scripts/generate_shakespeare.jl --temperature 0.5 --num_samples 1 --max_chars 512
```

Lower temperatures produce more conservative text; higher temperatures increase variety (and chaos).

### Example Shakespeare Output

After training the default model (357K params, 15 epochs, final loss ~1.75), the model generates text like this:

```
=== Sample 1 (256 chars) ===
KING HENRY VI:
My lord, the voics of heard a sone me father,
And thou dost the dost stard for steen,
And shall be stard the prince with me the world.

GLOUCESTER:
Was he some to the prince was thou say
To be a mort to this a send me
```

```
=== Sample 2 (256 chars) ===
SICINIUS:
He shall not be so with the come the man,
The city and the some the with the with him.

CORIOLANUS:
Sir, the sould of the do not to be so
To shall the people, and the bood and some
That sir to my lord,
```

It's not Shakespeare, but the model has clearly learned character names, dialogue structure, verse formatting, and English-like word patterns — all from a 357K-parameter model trained on a single CPU in about 15 minutes.

### Run Tests

```bash
julia --project=. test/runtests.jl
```

## Project Structure

```
microgpt_jl/
  src/
    MicroGPT.jl        # Module definition and exports
    tokenizer.jl        # Character-level tokenizer with BOS token
    model.jl            # GPT model, TransformerBlock, RMSNorm, KV cache
    train.jl            # Training loops and generation functions
    checkpoint.jl       # JLD2-based model save/load
  scripts/
    run.jl              # CLI: train on names dataset
    run_shakespeare.jl  # CLI: train on Shakespeare
    generate_shakespeare.jl  # CLI: generate from saved checkpoint
  test/
    runtests.jl         # Test runner
    test_tokenizer.jl   # Tokenizer unit tests
    test_model.jl       # Model component unit tests
    test_train.jl       # Training pipeline unit tests
    test_checkpoint.jl  # Checkpoint save/load tests
    test_integration.jl # End-to-end workflow tests
  eval/
    eval_julia.jl       # Benchmarking script for Python/Julia comparison
```

The source implementation is **786 lines** across 5 files, compared to Karpathy's ~200-line single-file original. The difference reflects the additional functionality — mini-batched Shakespeare training, KV-cached generation, checkpoint persistence, and CLI tooling — rather than verbosity. The test suite adds another **556 lines** across 6 files.

## Architecture

The model exactly matches Karpathy's Python microgpt:

| Component | Detail |
|-----------|--------|
| Normalization | RMSNorm (no learnable parameters) |
| Activation | Plain ReLU |
| Biases | None (all Dense layers are bias-free) |
| Output projection | Separate `lm_head` (no weight tying) |
| Embeddings | Token + positional (learned) |
| Attention | Multi-head with causal mask, KV cache for inference |

### Names model (matches Python exactly)
- `vocab_size=27, n_embd=16, n_layer=1, n_head=4, block_size=16`
- 4,192 parameters
- Trains in ~1 second on CPU

### Shakespeare model (default config)
- `vocab_size=66, n_embd=128, n_layer=4, n_head=4, block_size=256`
- 357,536 parameters
- Trains in ~15 minutes on CPU

## Performance Comparison (Names Task)

Both implementations trained on 32K names for 1,000 steps with identical hyperparameters:

| Metric | Python (microgpt.py) | Julia (MicroGPT.jl) |
|--------|---------------------|---------------------|
| Parameters | 4,192 | 4,192 |
| Training time | 89.2s | 1.3s |
| ms/step | 89.2 | 1.3 |
| Final loss | 2.65 | 2.30 |
| Inference (10 samples) | 0.34s | 0.33s |

Julia trains **~70x faster** than the pure-Python original on the names task. This is expected — Karpathy's microgpt uses no external libraries (no NumPy, no PyTorch), implementing autograd from scratch in pure Python. The Julia port uses Flux.jl which leverages optimized BLAS routines and compiled code.

## Credits

- **[Andrej Karpathy](https://karpathy.ai/)** for creating [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) — the original pure-Python GPT implementation that this project is based on. Also for [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and the [names dataset](https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt).
- **[Flux.jl](https://fluxml.ai/)** — the Julia ML framework used for this port.
- **[Claude Code](https://claude.ai/claude-code)** — Anthropic's AI coding agent, which pair-programmed the entire implementation.

## License

MIT License. See [LICENSE](LICENSE) for details.
