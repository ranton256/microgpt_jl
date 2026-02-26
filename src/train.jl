"""
Training loop and generation for MicroGPT.

- Online learning: one document per step (names workflow)
- Mini-batched epoch training with warmup + cosine decay (Shakespeare workflow)
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

# ============================================================================ #
# Shakespeare / continuous-text workflow
# ============================================================================ #

"""Download tiny-shakespeare.txt and return the raw text as a single String."""
function load_shakespeare(path::String="shakespeare.txt")
    if !isfile(path)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        download(url, path)
    end
    return read(path, String)
end

"""
Split a text into fixed-length token chunks for training.

Encodes text as character IDs (no BOS wrapping — chunks are arbitrary slices
of continuous text, not complete documents). Each chunk has `chunk_size + 1`
tokens (input + 1-shifted target). Drops the final short chunk.

Returns `Vector{Vector{Int}}`.
"""
function chunk_text(tokenizer::CharTokenizer, text::String, chunk_size::Int)
    # Encode full text as character IDs (no BOS wrapping)
    all_ids = Int[tokenizer.stoi[string(ch)] for ch in text]

    chunks = Vector{Vector{Int}}()
    for start in 1:chunk_size:(length(all_ids) - chunk_size)
        push!(chunks, all_ids[start:start + chunk_size])  # length = chunk_size + 1
    end
    return chunks
end

"""
Stack chunks into batched (input, target) matrix pairs.

Returns `Vector{Tuple{Matrix{Int}, Matrix{Int}}}` where each element is
`(input_batch, target_batch)` of size `(block_size, batch_size)`.
"""
function make_batches(chunks::Vector{Vector{Int}}, batch_size::Int)
    batches = Tuple{Matrix{Int}, Matrix{Int}}[]
    n = length(chunks)
    block_size = length(chunks[1]) - 1
    for start in 1:batch_size:n - batch_size + 1
        input_batch = Matrix{Int}(undef, block_size, batch_size)
        target_batch = Matrix{Int}(undef, block_size, batch_size)
        for (j, idx) in enumerate(start:start + batch_size - 1)
            input_batch[:, j] = chunks[idx][1:end-1]
            target_batch[:, j] = chunks[idx][2:end]
        end
        push!(batches, (input_batch, target_batch))
    end
    return batches
end

"""
Train the GPT model on pre-batched data with warmup + cosine LR decay.

Epoch-based training with shuffled batch order per epoch, periodic logging,
and optional sample generation at intervals.
"""
function train_batched!(model, batches::Vector{Tuple{Matrix{Int}, Matrix{Int}}};
                        device=cpu, num_epochs::Int=15, learning_rate::Float64=3e-4,
                        warmup_frac::Float64=0.1, seed::Int=42,
                        beta1::Float64=0.9, beta2::Float64=0.98,
                        log_interval::Int=10, sample_interval::Int=0,
                        tokenizer::Union{CharTokenizer, Nothing}=nothing)
    Random.seed!(seed)
    model = model |> device

    opt_state = Flux.setup(Adam(Float64(learning_rate), (beta1, beta2)), model)

    total_steps = num_epochs * length(batches)
    warmup_steps = round(Int, warmup_frac * total_steps)
    vocab_size = model.vocab_size
    step = 0
    losses_log = Float64[]

    @printf("total steps: %d (%d batches × %d epochs), warmup: %d steps\n",
            total_steps, length(batches), num_epochs, warmup_steps)
    flush(stdout)

    for epoch in 1:num_epochs
        batch_order = shuffle(1:length(batches))

        for batch_idx in batch_order
            step += 1
            input_ids, target_ids = batches[batch_idx]
            input_dev = input_ids |> device
            target_dev = target_ids |> device

            # LR schedule: linear warmup then cosine decay
            if step <= warmup_steps
                lr_t = learning_rate * (step / warmup_steps)
            else
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr_t = learning_rate * 0.5 * (1.0 + cos(Float64(π) * progress))
            end
            Flux.adjust!(opt_state, lr_t)

            # Forward + backward
            loss_val, grads = Flux.withgradient(model) do m
                logits = m(input_dev)  # (vocab_size, T, B)
                T_len = size(logits, 2)
                B_len = size(logits, 3)
                logits_2d = reshape(logits, vocab_size, T_len * B_len)
                targets_1d = reshape(target_dev, T_len * B_len)
                Flux.logitcrossentropy(logits_2d, Flux.onehotbatch(targets_1d, 1:vocab_size))
            end

            Flux.update!(opt_state, model, grads[1])
            push!(losses_log, Float64(loss_val))

            if step % log_interval == 0 || step == 1
                @printf("epoch %2d | step %5d / %5d | lr %.2e | loss %.4f\n",
                        epoch, step, total_steps, lr_t, loss_val)
                flush(stdout)
            end

            # Generate samples to show learning progress
            if sample_interval > 0 && tokenizer !== nothing && step % sample_interval == 0
                println("--- sample at step $step ---")
                generate_long(model, tokenizer; device=device, max_chars=min(200, model.block_size),
                              temperature=0.8f0, seed=step, num_samples=1)
                println("---")
            end
        end
    end

    return model, losses_log
end

"""
Generate continuous text from the trained model (for Shakespeare etc.).

Uses KV-cached autoregressive generation starting from BOS.
Does NOT stop on BOS (unlike `generate()` for names) since this is
continuous text, not delimited documents. Caps at `block_size` positions.
"""
function generate_long(model, tokenizer::CharTokenizer;
                       device=cpu, max_chars::Int=256,
                       temperature::Float32=0.8f0, seed::Int=42,
                       num_samples::Int=1)
    Random.seed!(seed)
    model = model |> device
    block_size = model.block_size
    vocab_size = model.vocab_size
    n_layer = length(model.blocks)
    bid = bos_id(tokenizer)

    effective_max = min(max_chars, block_size)

    for s in 1:num_samples
        cache = KVCache(n_layer)
        token_id = bid
        tokens = Int[]

        for pos in 1:effective_max
            logits = generate_step(model, token_id, pos, cache)
            scaled = logits ./ temperature
            probs = softmax(scaled)
            probs_vec = Float64.(vec(probs))
            probs_vec = max.(probs_vec, 0.0)
            total = sum(probs_vec)
            total <= 0 ? (probs_vec .= 1.0 / length(probs_vec)) : (probs_vec ./= total)
            next_token = sample(1:vocab_size, Weights(probs_vec))
            token_id = next_token
            push!(tokens, next_token)
        end

        text = decode(tokenizer, tokens)
        @printf("=== Sample %d (%d chars) ===\n%s\n\n", s, length(text), text)
    end
end
