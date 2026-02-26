@testset "Checkpoint Save/Load" begin
    @testset "Round-trip config" begin
        mktempdir() do dir
            path = joinpath(dir, "test.jld2")
            tok = CharTokenizer(["abcdefghij"])
            model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=2,
                         block_size=16, n_head=2)

            capture_out() do
                save_checkpoint(path, model, tok)
            end
            @test isfile(path)

            model2, tok2 = load_checkpoint(path)
            @test model2.vocab_size == model.vocab_size
            @test model2.n_embd == model.n_embd
            @test model2.block_size == model.block_size
            @test length(model2.blocks) == length(model.blocks)
            @test model2.blocks[1].n_head == model.blocks[1].n_head
        end
    end

    @testset "Weight equality" begin
        mktempdir() do dir
            path = joinpath(dir, "test.jld2")
            tok = CharTokenizer(["abcdefghij"])
            model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                         block_size=16, n_head=2)

            capture_out() do
                save_checkpoint(path, model, tok)
            end
            model2, _ = load_checkpoint(path)

            # Same input should produce same logits
            input = [1, 3, 5]
            logits1 = model(input)
            logits2 = model2(input)
            @test logits1 ≈ logits2
        end
    end

    @testset "Tokenizer equality" begin
        mktempdir() do dir
            path = joinpath(dir, "test.jld2")
            tok = CharTokenizer(["hello world"])
            model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                         block_size=16, n_head=2)

            capture_out() do
                save_checkpoint(path, model, tok)
            end
            _, tok2 = load_checkpoint(path)

            @test tok2.vocab_size == tok.vocab_size
            @test tok2.stoi == tok.stoi
            @test tok2.itos == tok.itos
            @test bos_id(tok2) == bos_id(tok)
            @test decode(tok2, encode(tok2, "hello")) == "hello"
        end
    end

    @testset "Generation consistency after load" begin
        mktempdir() do dir
            path = joinpath(dir, "test.jld2")
            tok = CharTokenizer(["abcdefghij"])
            model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                         block_size=16, n_head=2)

            # Generate before save
            out1 = capture_out() do
                generate_long(model, tok; max_chars=10, num_samples=1, seed=123)
            end

            capture_out() do
                save_checkpoint(path, model, tok)
            end
            model2, tok2 = load_checkpoint(path)

            # Generate after load with same seed
            out2 = capture_out() do
                generate_long(model2, tok2; max_chars=10, num_samples=1, seed=123)
            end

            @test out1 == out2
        end
    end

    @testset "Missing file error" begin
        @test_throws Exception load_checkpoint("/tmp/nonexistent_checkpoint_abc123.jld2")
    end

    @testset "Directory creation" begin
        mktempdir() do dir
            nested_path = joinpath(dir, "deep", "nested", "dir", "model.jld2")
            tok = CharTokenizer(["abc"])
            model = GPT(vocab_size=tok.vocab_size, n_embd=8, n_layer=1,
                         block_size=4, n_head=2)

            capture_out() do
                save_checkpoint(nested_path, model, tok)
            end
            @test isfile(nested_path)
        end
    end
end
