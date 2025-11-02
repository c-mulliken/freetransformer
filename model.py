import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CausalSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 1024, 1024)))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        attn_output = self.out_proj(attn_output)
        return attn_output
        
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
    def forward_with_condition(self, x, r):
        """
        x: (B, T, D)  — activations from first half of decoder
        r: (B, T, D)  — latent projection (BinaryDecoder output)
        """
        B, T, D = x.shape
        assert r.shape == x.shape, "r must match x in shape"

        # LayerNorm before attention
        normed_x = self.ln1(x)
        normed_r = self.ln1(r)

        # ---- Custom attention with conditioned K,V ----
        qkv = self.attn.qkv_proj(normed_x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to heads
        q = q.view(B, T, self.attn.num_heads, self.attn.head_dim).transpose(1, 2)
        k = (self.attn.qkv_proj(normed_x + normed_r)
             .view(B, T, 3, self.attn.num_heads, self.attn.head_dim)
             .permute(2, 0, 3, 1, 4))[1]  # only take K
        v = (self.attn.qkv_proj(normed_x + normed_r)
             .view(B, T, 3, self.attn.num_heads, self.attn.head_dim)
             .permute(2, 0, 3, 1, 4))[2]  # and V

        # Compute masked attention manually (reuse existing mask)
        attn_weights = (q @ k.transpose(-2, -1)) / (self.attn.head_dim ** 0.5)
        mask = self.attn.mask[:, :, :T, :T]
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.attn.dropout(attn_probs)

        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, T, D)
        attn_output = self.attn.out_proj(attn_output)

        # Residual + feedforward as usual
        x = x + attn_output
        x = x + self.ff(self.ln2(x))
        return x
    

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1024:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

class NonCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(NonCausalSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        attn_output = self.out_proj(attn_output)
        return attn_output
    

    
class FTBinaryMapper(nn.Module):
    def __init__(self, H):
        super(FTBinaryMapper, self).__init__()
        self.H = H
        self.register_buffer("powers", 2 ** torch.arange(H))

    def forward(self, logits):
        """
        logits: (B, T, H) - logits for each bit
        Returns: z_onehot (B, T, 2^H), kl_divergence (B, T)

        Implements Equation 6-8 from the paper.
        """
        B, T, H = logits.shape
        assert H == self.H, "Input feature dimension must match H"

        # Sample bits independently (Equation 6)
        p = torch.sigmoid(logits)  # P(B_t,h = 1)
        u = torch.rand_like(p)
        bits = (u < p).float()  # sampled bits B_t,h

        # Convert bits to index: d = 1 + sum_h 2^(h-1) * B_h,t
        idx = (bits * self.powers).sum(dim=-1).long()

        # Create one-hot vector Y_t,d (Equation 7)
        z_onehot = F.one_hot(idx, num_classes=2**self.H).float()

        # Compute gradient probabilities G_t,d for all possible values (used for gradient)
        # G_t,d = P(B_t = U(d-1)) where U(d) is binary encoding of d
        # This is exp(sum_h log P(B_t,h = U_h(d-1)))
        all_indices = torch.arange(2**self.H, device=logits.device)  # 0 to 2^H - 1
        binary_matrix = ((all_indices.unsqueeze(-1) >> torch.arange(H, device=logits.device)) & 1).float()  # (2^H, H)

        # For each position (B, T), compute G_t,d for all d
        # log P(B_t,h = b) = b * log(p) + (1-b) * log(1-p)
        log_p = torch.log(p + 1e-10)  # (B, T, H)
        log_1mp = torch.log(1 - p + 1e-10)  # (B, T, H)

        # Broadcast and compute log probabilities for all binary combinations
        # (B, T, 1, H) with binary_matrix (2^H, H)
        log_probs = binary_matrix.unsqueeze(0).unsqueeze(0) * log_p.unsqueeze(2) + \
                    (1 - binary_matrix.unsqueeze(0).unsqueeze(0)) * log_1mp.unsqueeze(2)
        G = torch.exp(log_probs.sum(dim=-1))  # (B, T, 2^H)

        # Gradient pass-through (Equation 8): Y_t,d + G_t,d - detach(G_t,d)
        z_onehot_grad = z_onehot + G - G.detach()

        # Compute KL divergence (Equation 4)
        # D_KL(Q(Z_t | S) || P(Z_t)) = H*log(2) + sum_{z=1}^{2^H} Q(Z=z|S) log Q(Z=z|S)
        # Q(Z = z | S) is stored in G_t,d
        kl_divergence = H * torch.log(torch.tensor(2.0, device=logits.device)) + \
                       (G * torch.log(G + 1e-10)).sum(dim=-1)  # (B, T)

        return z_onehot_grad, kl_divergence

class FTBinaryDecoder(nn.Module):
    def __init__(self, H, embed_dim):
        super().__init__()
        self.fc = nn.Linear(2 ** H, embed_dim)

    def forward(self, z_onehot):
        return self.fc(z_onehot)
    
    def sample_uniform(self, x):
        B, T, _ = x.shape
        num_states = self.fc.in_features
        idx = torch.randint(0, num_states, (B, T), device=x.device)
        return F.one_hot(idx, num_classes=num_states).float()
    
class NonCausalCrossAttention(nn.Module):
    """Cross-attention where Q comes from one source and K,V from another"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(NonCausalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_input, kv_input):
        """
        q_input: (B, T, D) - query input
        kv_input: (B, T, D) - key/value input
        """
        B, T, C = q_input.size()

        # Project queries from q_input
        q = self.q_proj(q_input).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Project keys and values from kv_input
        kv = self.kv_proj(kv_input).reshape(B, T, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Non-causal attention (no masking)
        attn_weights = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        attn_output = self.out_proj(attn_output)
        return attn_output


class FTEncoder(nn.Module):
    """
    Free Transformer encoder (§3.3 Fleuret 2025)
    Non-causal attention block that maps first-half decoder features → bit logits.
    Uses learned query ζ with cross-attention to decoder features.
    """
    def __init__(self, embed_dim, num_heads, H, dropout=0.1):
        super(FTEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.H = H

        # learned query token ζ replicated to match sequence length
        self.zeta = nn.Parameter(torch.randn(1, 1, embed_dim))

        # one non-causal cross-attention block
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = NonCausalCrossAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, embed_dim * 4, dropout)

        # projection head → H bit logits per token
        self.ln_readout = nn.LayerNorm(embed_dim)
        self.readout = nn.Linear(embed_dim, H)

    def forward(self, x):
        """
        x: (B, T, D) from halfway point of decoder
        returns: (B, T, H) latent bit logits
        """
        B, T, D = x.shape

        # replicate learned ζ query for each token position
        zeta = self.zeta.expand(B, T, D)

        # cross-attention: queries = ζ, keys/values = x
        attn_out = self.attn(self.ln_q(zeta), self.ln_kv(x))

        # feedforward with residual
        y = attn_out + self.ff(self.ln2(attn_out))

        # readout to bit logits
        logits = self.readout(self.ln_readout(y))  # (B, T, H)
        return logits

class FreeTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, H, dropout=0.1):
        super(FreeTransformer, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(max_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.ft_encoder = FTEncoder(embed_dim, num_heads, H, dropout)
        self.binary_mapper = FTBinaryMapper(H)
        self.binary_decoder = FTBinaryDecoder(H, embed_dim)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        half_layers = len(self.layers) // 2
        x = self.token_embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers[:half_layers]:
            x = layer(x)

        if self.training:
            enc_logits = self.ft_encoder(x)
            z_onehot, kl_bits = self.binary_mapper(enc_logits)
        else:
            z_onehot = self.binary_decoder.sample_uniform(x)
            kl_bits = None

        r = self.binary_decoder(z_onehot)
        x = self.layers[half_layers].forward_with_condition(x, r)
        for layer in self.layers[half_layers + 1:]:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, kl_bits

