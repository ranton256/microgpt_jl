"""
Character-level tokenizer with BOS and EOS tokens.

Builds a vocabulary from the dataset with special tokens at indices 1 and 2
(Julia is 1-indexed). Characters are sorted and assigned indices starting at 3.
"""
struct CharTokenizer
    stoi::Dict{String,Int}
    itos::Dict{Int,String}
    vocab_size::Int
end

const BOS_TOKEN = "<BOS>"
const EOS_TOKEN = "<EOS>"
const BOS_ID = 1
const EOS_ID = 2

function CharTokenizer(docs::Vector{String})
    # Collect unique characters across all documents, sorted
    chars = sort(collect(Set(join(docs))))
    stoi = Dict{String,Int}()
    itos = Dict{Int,String}()
    # Reserve indices 1 and 2 for special tokens
    stoi[BOS_TOKEN] = BOS_ID
    stoi[EOS_TOKEN] = EOS_ID
    itos[BOS_ID] = BOS_TOKEN
    itos[EOS_ID] = EOS_TOKEN
    for (i, ch) in enumerate(chars)
        idx = i + 2  # start after BOS=1, EOS=2
        stoi[string(ch)] = idx
        itos[idx] = string(ch)
    end
    vocab_size = length(chars) + 2  # chars + BOS + EOS
    CharTokenizer(stoi, itos, vocab_size)
end

function encode(tok::CharTokenizer, text::String)
    ids = Int[BOS_ID]
    for ch in text
        push!(ids, tok.stoi[string(ch)])
    end
    push!(ids, EOS_ID)
    return ids
end

function decode(tok::CharTokenizer, ids::Vector{Int})
    chars = String[]
    for id in ids
        id == BOS_ID && continue
        id == EOS_ID && break
        push!(chars, tok.itos[id])
    end
    return join(chars)
end
