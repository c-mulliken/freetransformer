from typing import List, Dict, Tuple

def char_tokenizer(texts: List[str]) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
    vocab = sorted(set("".join(texts)))
    stoi = {c:i for i,c in enumerate(vocab)}
    itos = {i:c for c,i in stoi.items()}
    enc = [[stoi[c] for c in t] for t in texts]
    return enc, stoi, itos
