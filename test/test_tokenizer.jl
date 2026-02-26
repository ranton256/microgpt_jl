@testset "CharTokenizer" begin
    @testset "Construction" begin
        tok = CharTokenizer(["hello", "world"])
        # Unique chars: d, e, h, l, o, r, w = 7 chars + BOS = 8
        @test tok.vocab_size == 8
        @test length(tok.stoi) == 8
        @test length(tok.itos) == 8
    end

    @testset "Character ordering" begin
        tok = CharTokenizer(["bac"])
        # Sorted unique chars: a=1, b=2, c=3, BOS=4
        @test tok.stoi["a"] == 1
        @test tok.stoi["b"] == 2
        @test tok.stoi["c"] == 3
        @test tok.stoi["<BOS>"] == 4
        @test tok.itos[1] == "a"
        @test tok.itos[2] == "b"
        @test tok.itos[3] == "c"
        @test tok.itos[4] == "<BOS>"
    end

    @testset "bos_id" begin
        tok = CharTokenizer(["abc"])
        @test bos_id(tok) == tok.vocab_size
        @test bos_id(tok) == 4  # a=1, b=2, c=3, BOS=4
    end

    @testset "encode" begin
        tok = CharTokenizer(["abc"])
        ids = encode(tok, "abc")
        bid = bos_id(tok)
        # Should be [BOS, a, b, c, BOS]
        @test length(ids) == 5
        @test ids[1] == bid
        @test ids[end] == bid
        @test ids[2] == tok.stoi["a"]
        @test ids[3] == tok.stoi["b"]
        @test ids[4] == tok.stoi["c"]
    end

    @testset "decode" begin
        tok = CharTokenizer(["abc"])
        ids = encode(tok, "abc")
        text = decode(tok, ids)
        @test text == "abc"
    end

    @testset "Round-trip" begin
        tok = CharTokenizer(["hello world", "foo bar"])
        for s in ["hello", "world", "foo", "bar", "h", "of"]
            @test decode(tok, encode(tok, s)) == s
        end
    end

    @testset "Empty string" begin
        tok = CharTokenizer(["abc"])
        ids = encode(tok, "")
        bid = bos_id(tok)
        @test ids == [bid, bid]
        @test decode(tok, ids) == ""
    end

    @testset "Single character" begin
        tok = CharTokenizer(["x"])
        @test tok.vocab_size == 2  # x + BOS
        @test decode(tok, encode(tok, "x")) == "x"
    end

    @testset "Multi-document vocab" begin
        tok = CharTokenizer(["abc", "def", "adf"])
        # Unique: a, b, c, d, e, f = 6 + BOS = 7
        @test tok.vocab_size == 7
    end
end
