"""
Instrumented evaluation of the Julia MicroGPT port.
Records per-step losses, total training time, and inference time.
Outputs results as JSON for comparison.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MicroGPT
using Random

const NUM_STEPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
const SEED = 42

# --- Dataset ---
docs = load_dataset()
Random.seed!(SEED)
shuffled_docs = shuffle(docs)
println("num docs: $(length(docs))")

# --- Tokenizer ---
tokenizer = CharTokenizer(docs)
println("vocab size: $(tokenizer.vocab_size)")

# --- Model ---
Random.seed!(SEED)
model = GPT(
    vocab_size=tokenizer.vocab_size,
    n_embd=16,
    n_layer=1,
    block_size=8,
    n_head=4,
)
num_params = count_parameters(model)
println("num params: $num_params")

device = get_device()

# --- Training (inline to capture per-step losses) ---
using Flux
using Flux: Adam, logitcrossentropy, onehotbatch, withgradient, setup, update!, adjust!
using Printf: @printf, @sprintf
using NNlib: softmax
using StatsBase: Weights, sample

learning_rate = 0.01
beta1, beta2 = 0.9, 0.95

model_dev = model |> device
opt_state = setup(Adam(learning_rate, (beta1, beta2)), model_dev)

block_size = model.block_size
vocab_size_m = model.vocab_size

losses_log = Float64[]

println("\n=== Training $NUM_STEPS steps ===")

# Warmup: 1 step to compile
let
    doc = shuffled_docs[1]
    tokens = encode(tokenizer, doc)
    n = min(block_size + 1, length(tokens))
    tokens = tokens[1:n]
    input_ids = tokens[1:end-1] |> device
    target_ids = tokens[2:end] |> device
    Flux.withgradient(model_dev) do m
        logits = m(input_ids)
        logits_2d = dropdims(logits; dims=3)
        logitcrossentropy(logits_2d, onehotbatch(target_ids, 1:vocab_size_m))
    end
end

# Reset model and optimizer for fair timing
Random.seed!(SEED)
model_dev = GPT(
    vocab_size=tokenizer.vocab_size,
    n_embd=16,
    n_layer=1,
    block_size=8,
    n_head=4,
) |> device
opt_state = setup(Adam(learning_rate, (beta1, beta2)), model_dev)

Random.seed!(SEED)
shuffled_docs = shuffle(docs)

train_start = time_ns()

for step in 1:NUM_STEPS
    doc = shuffled_docs[mod1(step, length(shuffled_docs))]
    tokens = encode(tokenizer, doc)
    n = min(block_size + 1, length(tokens))
    tokens = tokens[1:n]
    input_ids = tokens[1:end-1] |> device
    target_ids = tokens[2:end] |> device

    lr_t = learning_rate * (1.0 - (step - 1) / NUM_STEPS)
    adjust!(opt_state, lr_t)

    loss_val, grads = Flux.withgradient(model_dev) do m
        logits = m(input_ids)
        logits_2d = dropdims(logits; dims=3)
        logitcrossentropy(logits_2d, onehotbatch(target_ids, 1:vocab_size_m))
    end

    update!(opt_state, model_dev, grads[1])

    push!(losses_log, Float64(loss_val))
    if step % 100 == 0 || step == 1 || step == NUM_STEPS
        @printf("step %4d / %4d | loss %.4f\n", step, NUM_STEPS, loss_val)
    end
end

train_elapsed = (time_ns() - train_start) / 1e9
@printf("\nTraining time: %.2fs (%.1fms/step)\n", train_elapsed, train_elapsed / NUM_STEPS * 1000)

# --- Inference ---
NUM_SAMPLES = 10
temperature = 0.5f0
samples = String[]

println("\n=== Generating $NUM_SAMPLES samples ===")

n_layer = length(model_dev.blocks)

# Warmup: compile generate_step for all cache sizes (growing K/V tensors)
let
    warmup_cache = KVCache(n_layer)
    tid = MicroGPT.BOS_ID
    for p in 1:block_size
        generate_step(model_dev, tid, p, warmup_cache)
    end
end

Random.seed!(SEED)

infer_start = time_ns()
for i in 1:NUM_SAMPLES
    cache = KVCache(n_layer)
    token_id = MicroGPT.BOS_ID
    tokens = Int[MicroGPT.BOS_ID]
    for pos in 1:block_size
        logits = generate_step(model_dev, token_id, pos, cache)
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
        next_token = sample(1:vocab_size_m, Weights(probs_vec))
        next_token == MicroGPT.EOS_ID && break
        token_id = next_token
        push!(tokens, next_token)
    end
    name = decode(tokenizer, tokens)
    push!(samples, name)
    @printf("sample %2d: %s\n", i, name)
end

infer_elapsed = (time_ns() - infer_start) / 1e9
@printf("\nInference time: %.2fs (%.1fms/sample)\n", infer_elapsed, infer_elapsed / NUM_SAMPLES * 1000)

# --- Save results as JSON ---
results = Dict(
    "implementation" => "julia_flux",
    "num_steps" => NUM_STEPS,
    "num_params" => num_params,
    "vocab_size" => tokenizer.vocab_size,
    "block_size" => block_size,
    "n_embd" => 16,
    "n_layer" => 1,
    "n_head" => 4,
    "adam_betas" => [beta1, beta2],
    "init_std" => 0.02,
    "activation" => "squared_relu",
    "weight_tying" => true,
    "training_time_s" => train_elapsed,
    "ms_per_step" => train_elapsed / NUM_STEPS * 1000,
    "inference_time_s" => infer_elapsed,
    "ms_per_sample" => infer_elapsed / NUM_SAMPLES * 1000,
    "losses" => losses_log,
    "loss_first" => losses_log[1],
    "loss_last" => losses_log[end],
    "samples" => samples,
)

# Write JSON manually (avoid dependency)
open(joinpath(@__DIR__, "results_julia.json"), "w") do f
    write(f, "{\n")
    entries = []
    for (k, v) in sort(collect(results), by=first)
        if v isa String
            push!(entries, "  \"$k\": \"$v\"")
        elseif v isa Bool
            push!(entries, "  \"$k\": $(v ? "true" : "false")")
        elseif v isa Vector{Float64}
            vals = join([@sprintf("%.4f", x) for x in v], ", ")
            push!(entries, "  \"$k\": [$vals]")
        elseif v isa Vector
            vals = join(["\"$x\"" for x in v], ", ")
            push!(entries, "  \"$k\": [$vals]")
        elseif v isa AbstractFloat
            push!(entries, "  \"$k\": $(@sprintf("%.4f", v))")
        else
            push!(entries, "  \"$k\": $v")
        end
    end
    write(f, join(entries, ",\n"))
    write(f, "\n}\n")
end

println("\nResults saved to eval/results_julia.json")
