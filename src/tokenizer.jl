"""
Character-level tokenizer with a BOS token used for both start and end of sequence.

Builds a vocabulary from the dataset. Characters are sorted and assigned indices
starting at 1 (Julia is 1-indexed). The BOS token gets the last index, matching
the Python original where BOS = len(uchars).
"""
struct CharTokenizer
    stoi::Dict{String,Int}
    itos::Dict{Int,String}
    vocab_size::Int
end

const BOS_TOKEN = "<BOS>"

function CharTokenizer(docs::Vector{String})
    # Collect unique characters across all documents, sorted
    chars = sort(collect(Set(join(docs))))
    stoi = Dict{String,Int}()
    itos = Dict{Int,String}()
    # Characters get indices 1..n (matching Python's 0..n-1 but 1-indexed)
    for (i, ch) in enumerate(chars)
        stoi[string(ch)] = i
        itos[i] = string(ch)
    end
    # BOS gets the last index (matching Python: BOS = len(uchars))
    bos_id = length(chars) + 1
    stoi[BOS_TOKEN] = bos_id
    itos[bos_id] = BOS_TOKEN
    vocab_size = length(chars) + 1  # chars + BOS
    CharTokenizer(stoi, itos, vocab_size)
end

"""BOS token ID (last index in vocabulary)."""
bos_id(tok::CharTokenizer) = tok.stoi[BOS_TOKEN]

function encode(tok::CharTokenizer, text::String)
    bid = bos_id(tok)
    ids = Int[bid]
    for ch in text
        push!(ids, tok.stoi[string(ch)])
    end
    push!(ids, bid)  # BOS used as EOS too
    return ids
end

function decode(tok::CharTokenizer, ids::Vector{Int})
    bid = bos_id(tok)
    chars = String[]
    for id in ids
        id == bid && continue  # skip BOS at start; stop at BOS-as-EOS handled by caller
        push!(chars, tok.itos[id])
    end
    return join(chars)
end
