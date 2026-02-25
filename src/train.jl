"""
Training loop and generation for MicroGPT.

- Online learning: one document per step
- Adam optimizer with linear LR warmdown
- Autoregressive generation with temperature sampling
"""

using Downloads: download

"""Download the names dataset if not already cached."""
function load_dataset(path::String="input.txt")
    if !isfile(path)
        url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        download(url, path)
    end
    docs = String[]
    for line in eachline(path)
        s = strip(line)
        !isempty(s) && push!(docs, s)
    end
    return docs
end

"""
Train the GPT model on the dataset.

Arguments:
- `model`: GPT model
- `tokenizer`: CharTokenizer
- `docs`: Vector{String} of training documents
- `device`: cpu or gpu device function
- `num_steps`, `learning_rate`, `seed`: hyperparameters
"""
function train!(model, tokenizer::CharTokenizer, docs::Vector{String};
                device=cpu, num_steps::Int=1000, learning_rate::Float64=0.01,
                seed::Int=42, beta1::Float64=0.85, beta2::Float64=0.99)
    Random.seed!(seed)
    shuffled_docs = shuffle(docs)

    model = model |> device

    # Set up Adam optimizer
    opt_state = Flux.setup(Adam(Float64(learning_rate), (beta1, beta2)), model)

    block_size = model.block_size

    for step in 1:num_steps
        doc = shuffled_docs[mod1(step, length(shuffled_docs))]

        # Tokenize: [BOS, chars..., EOS]
        tokens = encode(tokenizer, doc)

        # Crop to block_size + 1 (input + target)
        n = min(block_size + 1, length(tokens))
        tokens = tokens[1:n]

        input_ids = tokens[1:end-1]   # (T,)
        target_ids = tokens[2:end]    # (T,)

        # Move to device
        input_ids_dev = input_ids |> device
        target_ids_dev = target_ids |> device

        # Linear LR warmdown
        lr_t = learning_rate * (1.0 - (step - 1) / num_steps)
        Flux.adjust!(opt_state, lr_t)

        # Forward + backward
        loss_val, grads = Flux.withgradient(model) do m
            logits = m(input_ids_dev)  # (vocab_size, T, 1)
            logits_2d = dropdims(logits; dims=3)  # (vocab_size, T)
            Flux.logitcrossentropy(logits_2d, Flux.onehotbatch(target_ids_dev, 1:m.vocab_size))
        end

        # Update parameters
        Flux.update!(opt_state, model, grads[1])

        # Print progress
        @printf("step %4d / %4d | loss %.4f\n", step, num_steps, loss_val)
    end

    return model
end

"""
Generate text autoregressively from the trained model using KV cache.

Uses temperature sampling. Generates until EOS token or block_size is reached.
Each step processes only the new token, reusing cached K/V from previous steps.
"""
function generate(model, tokenizer::CharTokenizer;
                  device=cpu, num_samples::Int=5, temperature::Float32=0.5f0,
                  seed::Int=42)
    Random.seed!(seed)
    model = model |> device
    block_size = model.block_size
    vocab_size = model.vocab_size
    n_layer = length(model.blocks)

    println("\n--- inference (new, hallucinated names) ---")

    bid = bos_id(tokenizer)

    for i in 1:num_samples
        cache = KVCache(n_layer)
        token_id = bid
        tokens = Int[bid]

        for pos in 1:block_size
            logits = generate_step(model, token_id, pos, cache)
            # Apply temperature
            scaled = logits ./ temperature
            probs = softmax(scaled)
            probs_vec = Float64.(vec(probs))
            probs_vec = max.(probs_vec, 0.0)
            total = sum(probs_vec)
            if total <= 0
                probs_vec .= 1.0 / length(probs_vec)
            else
                probs_vec ./= total
            end
            # Sample next token
            next_token = sample(1:vocab_size, Weights(probs_vec))
            next_token == bid && break  # BOS used as EOS
            token_id = next_token
            push!(tokens, next_token)
        end
        name = decode(tokenizer, tokens)
        @printf("sample %2d: %s\n", i, name)
    end
end
