def char_tokenizer(texts: list[str]) -> tuple[list[list[int]], dict[str, int], dict[int, str]]:
    vocab = sorted(set("".join(texts)))
    stoi = {c:i for i,c in enumerate(vocab)}
    itos = {i:c for c,i in stoi.items()}
    enc = [[stoi[c] for c in t] for t in texts]
    return enc, stoi, itos
