"""
GPT model definition using Flux.jl.

Decoder-only transformer following GPT-2 architecture with:
- RMSNorm (no learnable parameters) instead of LayerNorm
- Squared ReLU activation instead of GeLU
- No biases anywhere
- Weight tying between token embeddings and output projection
"""

# ---------------------------------------------------------------------------- #
# Custom layers
# ---------------------------------------------------------------------------- #

"""
RMSNorm without learnable parameters.
`scale = (mean(x²) + ε)^(-0.5); out = x * scale`
Operates along the first (feature) dimension.
"""
struct RMSNorm
    eps::Float32
end

RMSNorm() = RMSNorm(1f-5)
Flux.@layer RMSNorm trainable=()

function (m::RMSNorm)(x::AbstractArray)
    # x shape: (n_embd, seq_len...) — norm along dim 1
    ms = mean(x .^ 2; dims=1)
    scale = (ms .+ m.eps) .^ (-0.5f0)
    return x .* scale
end

"""Squared ReLU activation: relu(x)^2"""
squared_relu(x) = NNlib.relu(x) .^ 2

# ---------------------------------------------------------------------------- #
# Transformer block
# ---------------------------------------------------------------------------- #

struct TransformerBlock
    norm1::RMSNorm
    attn_wq::Dense
    attn_wk::Dense
    attn_wv::Dense
    attn_wo::Dense
    norm2::RMSNorm
    mlp_fc1::Dense
    mlp_fc2::Dense
    n_head::Int
    head_dim::Int
end

Flux.@layer TransformerBlock trainable=(norm1, attn_wq, attn_wk, attn_wv, attn_wo, norm2, mlp_fc1, mlp_fc2)

function TransformerBlock(n_embd::Int, n_head::Int; init_std=0.02f0)
    head_dim = n_embd ÷ n_head
    init = (dims...) -> randn(Float32, dims...) .* init_std
    zero_init = (dims...) -> zeros(Float32, dims...)
    TransformerBlock(
        RMSNorm(),
        Dense(n_embd => n_embd, bias=false; init=init),
        Dense(n_embd => n_embd, bias=false; init=init),
        Dense(n_embd => n_embd, bias=false; init=init),
        Dense(n_embd => n_embd, bias=false; init=zero_init),  # zero init
        RMSNorm(),
        Dense(n_embd => 4*n_embd, bias=false; init=init),
        Dense(4*n_embd => n_embd, bias=false; init=zero_init),  # zero init
        n_head,
        head_dim,
    )
end

function (blk::TransformerBlock)(x::AbstractArray, mask::AbstractArray)
    # x: (n_embd, T, B), mask: (T, T)
    n_embd = size(x, 1)
    T = size(x, 2)
    B = size(x, 3)
    nh = blk.n_head
    hd = blk.head_dim

    # --- Multi-head self-attention ---
    residual = x
    x_norm = blk.norm1(x)  # (n_embd, T, B)

    q = blk.attn_wq(x_norm)  # (n_embd, T, B)
    k = blk.attn_wk(x_norm)
    v = blk.attn_wv(x_norm)

    # Reshape to (head_dim, T, n_head, B) then to (head_dim, T, n_head*B) for batched_mul
    q = reshape(q, hd, nh, T, B)
    k = reshape(k, hd, nh, T, B)
    v = reshape(v, hd, nh, T, B)
    q = permutedims(q, (1, 3, 2, 4))  # (hd, T, nh, B)
    k = permutedims(k, (1, 3, 2, 4))
    v = permutedims(v, (1, 3, 2, 4))
    q = reshape(q, hd, T, nh * B)
    k = reshape(k, hd, T, nh * B)
    v = reshape(v, hd, T, nh * B)

    # Attention scores: (T, T, nh*B) = k' * q / sqrt(hd)
    scale = Float32(1.0 / sqrt(Float64(hd)))
    attn = NNlib.batched_mul(permutedims(k, (2, 1, 3)), q) .* scale  # (T, T, nh*B)

    # Apply causal mask: mask is (T, T), true where allowed
    # Set masked positions to -Inf
    attn = attn .+ ifelse.(mask, 0f0, Float32(-Inf))

    attn = NNlib.softmax(attn; dims=1)  # softmax over key dimension (dim 1)

    # Weighted sum of values: (hd, T, nh*B)
    out = NNlib.batched_mul(v, attn)  # (hd, T, nh*B)

    # Reshape back to (n_embd, T, B)
    out = reshape(out, hd, T, nh, B)
    out = permutedims(out, (1, 3, 2, 4))  # (hd, nh, T, B)
    out = reshape(out, n_embd, T, B)

    x = blk.attn_wo(out) .+ residual

    # --- MLP ---
    residual = x
    x_norm = blk.norm2(x)
    x = blk.mlp_fc1(x_norm)
    x = squared_relu(x)
    x = blk.mlp_fc2(x)
    x = x .+ residual

    return x
end

# ---------------------------------------------------------------------------- #
# GPT model
# ---------------------------------------------------------------------------- #

struct GPT
    wte::Embedding      # token embeddings (vocab_size, n_embd)
    wpe::Embedding      # position embeddings (block_size, n_embd)
    norm0::RMSNorm      # pre-transformer norm (applied after embeddings)
    blocks::Vector{TransformerBlock}
    vocab_size::Int
    block_size::Int
    n_embd::Int
end

Flux.@layer GPT trainable=(wte, wpe, norm0, blocks)

function GPT(; vocab_size::Int, n_embd::Int=16, n_layer::Int=1,
               block_size::Int=8, n_head::Int=4, init_std::Float32=0.02f0)
    init = (dims...) -> randn(Float32, dims...) .* init_std
    wte = Embedding(vocab_size => n_embd; init=init)
    wpe = Embedding(block_size => n_embd; init=init)
    norm0 = RMSNorm()
    blocks = [TransformerBlock(n_embd, n_head; init_std=init_std) for _ in 1:n_layer]
    GPT(wte, wpe, norm0, blocks, vocab_size, block_size, n_embd)
end

function (m::GPT)(token_ids::AbstractArray{<:Integer})
    # token_ids: (T,) or (T, B) — integer token indices (1-indexed)
    if ndims(token_ids) == 1
        token_ids = reshape(token_ids, :, 1)
    end
    T = size(token_ids, 1)
    @assert T <= m.block_size "Sequence length $T exceeds block_size $(m.block_size)"

    # Embeddings: Flux Embedding expects indices, returns (n_embd, length)
    tok_emb = m.wte(token_ids)   # (n_embd, T, B)
    pos_ids = collect(1:T)       # (T,) — same positions for all batches
    pos_emb = m.wpe(pos_ids)     # (n_embd, T)

    x = tok_emb .+ pos_emb      # broadcast pos_emb across batch dim
    x = m.norm0(x)

    # Causal mask: lower triangular (T, T)
    mask = _causal_mask(T, token_ids)

    for blk in m.blocks
        x = blk(x, mask)
    end

    # Output projection via weight tying with wte
    # wte.weight is (n_embd, vocab_size), so wte.weight' is (vocab_size, n_embd)
    # logits = wte.weight' * x  →  (V, T, B)
    W = m.wte.weight'  # (vocab_size, n_embd)
    logits = NNlib.batched_mul(
        reshape(W, m.vocab_size, m.n_embd, 1),  # (V, E, 1) — broadcast over batch
        x  # (E, T, B)
    )  # (V, T, B)

    return logits
end

"""Build a causal (lower-triangular) mask on the same device as token_ids."""
function _causal_mask(T::Int, token_ids::AbstractArray)
    # Returns a Bool matrix (T, T) that is true where attention is allowed
    # mask[i,j] = true means position j can attend to position i (i <= j is causal)
    # Since our attn scores are (key_pos, query_pos), we want key_pos <= query_pos
    mask = [i <= j for i in 1:T, j in 1:T]
    return mask
end

# ---------------------------------------------------------------------------- #
# KV cache for autoregressive generation
# ---------------------------------------------------------------------------- #

"""
Per-layer KV cache for efficient autoregressive generation.
Each layer stores projected K and V tensors: (n_embd, T_cached, B).
"""
mutable struct KVCache
    layers::Vector{Tuple{Union{Nothing,AbstractArray}, Union{Nothing,AbstractArray}}}
end

KVCache(n_layer::Int) = KVCache([(nothing, nothing) for _ in 1:n_layer])

"""
Single-token forward pass through a TransformerBlock with KV cache.

- `x`: (n_embd, 1, B) — the new token's hidden state
- `k_cached`, `v_cached`: (n_embd, T_prev, B) or `nothing`

Returns `(x_out, k_all, v_all)` where k_all/v_all include the new token.
"""
function forward_cached(blk::TransformerBlock, x::AbstractArray, k_cached, v_cached)
    n_embd = size(x, 1)
    B = size(x, 3)
    nh = blk.n_head
    hd = blk.head_dim

    # --- Multi-head self-attention with KV cache ---
    residual = x
    x_norm = blk.norm1(x)  # (n_embd, 1, B)

    q_new = blk.attn_wq(x_norm)  # (n_embd, 1, B)
    k_new = blk.attn_wk(x_norm)  # (n_embd, 1, B)
    v_new = blk.attn_wv(x_norm)  # (n_embd, 1, B)

    # Append new K, V to cache
    k_all = k_cached === nothing ? k_new : cat(k_cached, k_new; dims=2)
    v_all = v_cached === nothing ? v_new : cat(v_cached, v_new; dims=2)

    T_kv = size(k_all, 2)  # total cached length including new token

    # Reshape for multi-head: q is (hd, 1, nh*B), k/v are (hd, T_kv, nh*B)
    q = reshape(reshape(q_new, hd, nh, 1, B), hd, 1, nh * B)
    k = reshape(permutedims(reshape(k_all, hd, nh, T_kv, B), (1, 3, 2, 4)), hd, T_kv, nh * B)
    v = reshape(permutedims(reshape(v_all, hd, nh, T_kv, B), (1, 3, 2, 4)), hd, T_kv, nh * B)

    # Attention: (T_kv, 1, nh*B) = k' * q / sqrt(hd)
    scale = Float32(1.0 / sqrt(Float64(hd)))
    attn = NNlib.batched_mul(permutedims(k, (2, 1, 3)), q) .* scale  # (T_kv, 1, nh*B)
    # No causal mask needed — q is the latest position, attends to all cached positions
    attn = NNlib.softmax(attn; dims=1)

    # Weighted sum: (hd, 1, nh*B)
    out = NNlib.batched_mul(v, attn)  # (hd, 1, nh*B)

    # Reshape back to (n_embd, 1, B)
    out = reshape(permutedims(reshape(out, hd, 1, nh, B), (1, 3, 2, 4)), n_embd, 1, B)

    x = blk.attn_wo(out) .+ residual

    # --- MLP ---
    residual = x
    x_norm = blk.norm2(x)
    x = blk.mlp_fc1(x_norm)
    x = squared_relu(x)
    x = blk.mlp_fc2(x)
    x = x .+ residual

    return x, k_all, v_all
end

"""
Single-token forward pass through the GPT model with KV cache.

- `token_id`: integer token index (1-indexed)
- `pos`: position index (1-indexed)
- `cache`: KVCache — updated in-place

Returns logits vector of shape `(vocab_size,)`.
"""
function generate_step(m::GPT, token_id::Int, pos::Int, cache::KVCache)
    tok_emb = m.wte([token_id])   # (n_embd, 1)
    pos_emb = m.wpe([pos])        # (n_embd, 1)
    x = tok_emb .+ pos_emb        # (n_embd, 1)
    x = reshape(x, :, 1, 1)       # (n_embd, 1, 1)
    x = m.norm0(x)

    for (i, blk) in enumerate(m.blocks)
        k_cached, v_cached = cache.layers[i]
        x, k_new, v_new = forward_cached(blk, x, k_cached, v_cached)
        cache.layers[i] = (k_new, v_new)
    end

    # Output projection via weight tying
    logits = m.wte.weight' * x[:, 1, 1]  # (vocab_size,)
    return logits
end

# ---------------------------------------------------------------------------- #
# Utilities
# ---------------------------------------------------------------------------- #

"""Count the number of trainable parameters."""
function count_parameters(m)
    total = Ref(0)
    Flux.fmap(m; exclude=x -> x isa AbstractArray{<:Number}) do x
        total[] += length(x)
        x
    end
    return total[]
end
