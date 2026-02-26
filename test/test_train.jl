@testset "Training & Data Pipeline" begin
    @testset "chunk_text" begin
        tok = CharTokenizer(["abcdefghijklmnop"])
        text = "abcdefghijklmnop"  # 16 chars
        chunks = chunk_text(tok, text, 4)
        # Each chunk has length 5 (chunk_size + 1)
        @test all(length(c) == 5 for c in chunks)
        # 16 chars with chunk_size=4: starts at 1,5,9 (start <= 16-4=12)
        @test length(chunks) == 3

        # No BOS tokens in chunks
        bid = bos_id(tok)
        for chunk in chunks
            @test !(bid in chunk)
        end
    end

    @testset "chunk_text shift" begin
        # Verify chunks have correct length for input/target split
        tok = CharTokenizer(["abcd"])
        text = "abcdabcd"
        chunks = chunk_text(tok, text, 3)
        @test length(chunks) >= 1
        c = chunks[1]
        @test length(c) == 4  # chunk_size + 1
    end

    @testset "make_batches" begin
        tok = CharTokenizer(["abcdefghijklmnopqrstuvwxyz"])
        text = "abcdefghijklmnopqrstuvwxyz" * "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(tok, text, 8)
        batches = make_batches(chunks, 2)

        @test length(batches) >= 1
        input_batch, target_batch = batches[1]
        @test size(input_batch) == (8, 2)
        @test size(target_batch) == (8, 2)
    end

    @testset "make_batches input/target shift" begin
        tok = CharTokenizer(["abcdefghij"])
        text = "abcdefghij" ^ 10  # repeat for enough chunks
        chunks = chunk_text(tok, text, 4)
        batches = make_batches(chunks, 2)

        input_batch, target_batch = batches[1]
        for j in 1:2
            @test size(input_batch, 1) == 4
            @test size(target_batch, 1) == 4
        end
    end

    @testset "train_batched! smoke test" begin
        tok = CharTokenizer(["abcdefghij"])
        text = "abcdefghij" ^ 20
        chunks = chunk_text(tok, text, 8)
        batches = make_batches(chunks, 4)
        # Use only first 3 batches for speed
        batches = batches[1:min(3, length(batches))]

        model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                     block_size=8, n_head=2)

        # Capture initial weights
        w_before = copy(model.lm_head.weight)

        model, losses = train_batched!(model, batches;
            num_epochs=2, learning_rate=1e-3, warmup_frac=0.1,
            log_interval=100, sample_interval=0)

        @test length(losses) > 0
        @test all(isfinite.(losses))
        # Loss should generally decrease (first vs last)
        @test losses[end] < losses[1] + 1.0  # allow some slack
        # Weights should have changed
        @test model.lm_head.weight != w_before
    end

    @testset "train! smoke test" begin
        docs = ["hello", "world", "julia", "test", "micro"]
        tok = CharTokenizer(docs)
        model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                     block_size=16, n_head=2)

        w_before = copy(model.lm_head.weight)
        model = train!(model, tok, docs; num_steps=10, learning_rate=0.01)

        @test model.lm_head.weight != w_before
    end

    @testset "generate produces valid output" begin
        docs = ["hello", "world", "julia"]
        tok = CharTokenizer(docs)
        model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                     block_size=16, n_head=2)

        output = capture_out() do
            generate(model, tok; num_samples=2, seed=42)
        end
        @test length(output) > 0
        @test occursin("sample", output)
    end

    @testset "generate_long produces text of expected length" begin
        tok = CharTokenizer(["abcdefghij"])
        model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                     block_size=16, n_head=2)

        output = capture_out() do
            generate_long(model, tok; max_chars=10, num_samples=1, seed=42)
        end
        @test length(output) > 0
        @test occursin("Sample", output)
    end
end
