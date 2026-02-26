@testset "Model Components" begin
    @testset "RMSNorm" begin
        norm = RMSNorm()
        x = randn(Float32, 8, 4, 2)
        y = norm(x)
        @test size(y) == size(x)
        @test all(isfinite.(y))
    end

    @testset "RMSNorm zero input" begin
        norm = RMSNorm()
        x = zeros(Float32, 8, 4, 2)
        y = norm(x)
        @test all(isfinite.(y))
        @test all(y .== 0f0)
    end

    @testset "TransformerBlock shape" begin
        n_embd = 8
        n_head = 2
        blk = MicroGPT.TransformerBlock(n_embd, n_head)

        for (T, B) in [(4, 1), (8, 2), (1, 1)]
            x = randn(Float32, n_embd, T, B)
            mask = [i <= j for i in 1:T, j in 1:T]
            y = blk(x, mask)
            @test size(y) == (n_embd, T, B)
            @test all(isfinite.(y))
        end
    end

    @testset "TransformerBlock residual" begin
        n_embd = 8
        n_head = 2
        blk = MicroGPT.TransformerBlock(n_embd, n_head)
        x = randn(Float32, n_embd, 4, 1)
        mask = [i <= j for i in 1:4, j in 1:4]
        y = blk(x, mask)
        # Output should not be all zeros thanks to residual connections
        @test any(y .!= 0f0)
    end

    @testset "GPT constructor" begin
        model = GPT(vocab_size=10, n_embd=8, n_layer=2, block_size=16, n_head=2)
        @test model.vocab_size == 10
        @test model.block_size == 16
        @test model.n_embd == 8
        @test length(model.blocks) == 2
    end

    @testset "GPT forward pass - unbatched" begin
        model = GPT(vocab_size=10, n_embd=8, n_layer=1, block_size=16, n_head=2)
        input = [1, 3, 5, 7]  # (T=4,)
        logits = model(input)
        @test size(logits) == (10, 4, 1)
        @test all(isfinite.(logits))
    end

    @testset "GPT forward pass - batched" begin
        model = GPT(vocab_size=10, n_embd=8, n_layer=1, block_size=16, n_head=2)
        input = [1 2; 3 4; 5 6; 7 8]  # (T=4, B=2)
        logits = model(input)
        @test size(logits) == (10, 4, 2)
        @test all(isfinite.(logits))
    end

    @testset "GPT block_size assertion" begin
        model = GPT(vocab_size=10, n_embd=8, n_layer=1, block_size=4, n_head=2)
        # Sequence of length 5 > block_size=4 should error
        @test_throws AssertionError model([1, 2, 3, 4, 5])
    end

    @testset "Causal mask" begin
        ids = [1, 2, 3]
        mask = MicroGPT._causal_mask(3, ids)
        @test size(mask) == (3, 3)
        # mask[i,j] = (i <= j): key position i can attend to query position j
        @test mask[1, 1] == true
        @test mask[1, 2] == true
        @test mask[2, 1] == false
        @test mask[3, 3] == true
        @test mask[3, 1] == false
        @test mask[3, 2] == false
    end

    @testset "count_parameters" begin
        model = GPT(vocab_size=10, n_embd=8, n_layer=1, block_size=4, n_head=2)
        n = count_parameters(model)
        @test n > 0
        @test isa(n, Integer)
        # wte: 10*8=80, wpe: 4*8=32, lm_head: 8*10=80 (no bias)
        # Block: wq,wk,wv,wo each 8*8=64 (=256), mlp_fc1: 8*32=256, mlp_fc2: 32*8=256
        # Total block: 256+256+256=768. RMSNorm: 0 params.
        # Total: 80+32+768+80 = 960
        @test n == 960
    end

    @testset "KVCache" begin
        cache = KVCache(3)
        @test length(cache.layers) == 3
        @test all(layer -> layer == (nothing, nothing), cache.layers)
    end

    @testset "generate_step" begin
        model = GPT(vocab_size=10, n_embd=8, n_layer=2, block_size=16, n_head=2)
        cache = KVCache(2)

        # First step
        logits = generate_step(model, 1, 1, cache)
        @test length(logits) == 10
        @test all(isfinite.(logits))
        # Cache should now have arrays (not nothing)
        @test cache.layers[1][1] !== nothing
        @test cache.layers[1][2] !== nothing

        # Second step — cache should grow
        logits2 = generate_step(model, 3, 2, cache)
        @test length(logits2) == 10
        @test size(cache.layers[1][1], 2) == 2  # K now has 2 positions
    end

    @testset "forward_cached shape" begin
        n_embd = 8
        n_head = 2
        blk = MicroGPT.TransformerBlock(n_embd, n_head)
        x = randn(Float32, n_embd, 1, 1)

        # First call — no cache
        y, k, v = MicroGPT.forward_cached(blk, x, nothing, nothing)
        @test size(y) == (n_embd, 1, 1)
        @test size(k) == (n_embd, 1, 1)
        @test size(v) == (n_embd, 1, 1)

        # Second call — with cache
        x2 = randn(Float32, n_embd, 1, 1)
        y2, k2, v2 = MicroGPT.forward_cached(blk, x2, k, v)
        @test size(y2) == (n_embd, 1, 1)
        @test size(k2) == (n_embd, 2, 1)  # grew by 1
        @test size(v2) == (n_embd, 2, 1)
    end
end
