"""
Checkpoint save/load for MicroGPT models using JLD2.

Saves model config, Flux parameter state, and tokenizer to a single .jld2 file.
Always saves weights on CPU regardless of training device.
"""

using JLD2

"""
    save_checkpoint(path, model, tokenizer)

Save a trained model and its tokenizer to a JLD2 checkpoint file.

The checkpoint contains:
- `config`: Dict with GPT constructor arguments (vocab_size, n_embd, n_layer, block_size, n_head)
- `model_state`: Flux.state(model) on CPU
- `tokenizer`: Dict with stoi, itos, vocab_size
"""
function save_checkpoint(path::String, model::GPT, tokenizer::CharTokenizer)
    # Extract config from the live model
    config = Dict{String,Int}(
        "vocab_size" => model.vocab_size,
        "n_embd"     => model.n_embd,
        "n_layer"    => length(model.blocks),
        "block_size" => model.block_size,
        "n_head"     => model.blocks[1].n_head,
    )

    # Always save weights on CPU
    model_cpu = model |> cpu
    state = Flux.state(model_cpu)

    # Tokenizer as plain Dict (avoid serializing the struct directly)
    tok_data = Dict{String,Any}(
        "stoi"       => tokenizer.stoi,
        "itos"       => tokenizer.itos,
        "vocab_size" => tokenizer.vocab_size,
    )

    # Ensure parent directory exists
    mkpath(dirname(abspath(path)))

    jldsave(path;
        config      = config,
        model_state = state,
        tokenizer   = tok_data,
    )
    @printf("Checkpoint saved to %s\n", path)
end

"""
    load_checkpoint(path) -> (model, tokenizer)

Load a GPT model and CharTokenizer from a JLD2 checkpoint file.
Returns the model on CPU; caller is responsible for moving to device.
"""
function load_checkpoint(path::String)
    data = JLD2.load(path)

    config = data["config"]
    model = GPT(
        vocab_size = config["vocab_size"],
        n_embd     = config["n_embd"],
        n_layer    = config["n_layer"],
        block_size = config["block_size"],
        n_head     = config["n_head"],
    )
    Flux.loadmodel!(model, data["model_state"])

    tok_data = data["tokenizer"]
    tokenizer = CharTokenizer(tok_data["stoi"], tok_data["itos"], tok_data["vocab_size"])

    return model, tokenizer
end
