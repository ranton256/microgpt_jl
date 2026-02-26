@testset "Integration Tests" begin
    @testset "Names workflow end-to-end" begin
        docs = ["alice", "bob", "carol", "dave", "eve"]
        tok = CharTokenizer(docs)
        model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                     block_size=16, n_head=2)

        # Train
        model = train!(model, tok, docs; num_steps=10, learning_rate=0.01)

        # Generate
        output = capture_out() do
            generate(model, tok; num_samples=2, seed=42)
        end
        @test length(output) > 0
        @test occursin("sample", output)
    end

    @testset "Shakespeare workflow end-to-end (mini)" begin
        # Use a small synthetic text instead of downloading
        text = "To be, or not to be, that is the question. " ^50
        tok = CharTokenizer([text])
        chunks = chunk_text(tok, text, 16)
        @test length(chunks) > 0

        batches = make_batches(chunks, 4)
        batches = batches[1:min(3, length(batches))]
        @test length(batches) >= 1

        model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                     block_size=16, n_head=2)

        model, losses = train_batched!(model, batches;
            num_epochs=1, learning_rate=1e-3, warmup_frac=0.1,
            log_interval=100, sample_interval=0)

        @test all(isfinite.(losses))

        # Generate
        output = capture_out() do
            generate_long(model, tok; max_chars=16, num_samples=1, seed=42)
        end
        @test length(output) > 0
    end

    @testset "Train → save → load → generate" begin
        mktempdir() do dir
            # Train
            text = "Hello world, this is a test of the MicroGPT system. " ^ 30
            tok = CharTokenizer([text])
            chunks = chunk_text(tok, text, 16)
            batches = make_batches(chunks, 4)
            batches = batches[1:min(3, length(batches))]

            model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                         block_size=16, n_head=2)
            model, losses = train_batched!(model, batches;
                num_epochs=2, learning_rate=1e-3, warmup_frac=0.1,
                log_interval=100, sample_interval=0)

            # Save
            ckpt_path = joinpath(dir, "model.jld2")
            capture_out() do
                save_checkpoint(ckpt_path, model, tok)
            end
            @test isfile(ckpt_path)

            # Load
            model2, tok2 = load_checkpoint(ckpt_path)
            @test model2.vocab_size == tok.vocab_size

            # Generate from loaded model
            output = capture_out() do
                generate_long(model2, tok2; max_chars=16, num_samples=1, seed=42)
            end
            @test length(output) > 0
            @test occursin("Sample", output)
        end
    end

    @testset "get_device" begin
        device = MicroGPT.get_device()
        # Should return either cpu or gpu without error
        @test device in [Flux.cpu, Flux.gpu]
    end
end
