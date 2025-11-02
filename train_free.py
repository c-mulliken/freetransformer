import torch
import math

from model import FreeTransformer
from utils import char_tokenizer
from dataset import generate_synthetic_data

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic data
    data = generate_synthetic_data(10000)

    # Tokenize data
    enc, stoi, itos = char_tokenizer(data)

    # Hyperparameters
    vocab_size = len(stoi)
    max_len = 64 + 2  # 64 chars + 2 for prompt
    embed_dim = 128
    num_heads = 4
    ff_dim = 512
    num_layers = 4
    dropout = 0.1
    H = 16  # Paper uses H=16 for 2^16 = 65,536 latent states
    batch_size = 32
    num_epochs = 20
    lr = 1e-3
    kappa = math.log(2) / 2  # Free-bits threshold: 1/2 bit per token (best from paper)

    # Initialize model, loss function, and optimizer
    model = FreeTransformer(
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        H,
        dropout,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_ce_loss = 0
        total_kl_loss = 0

        for i in range(0, len(enc), batch_size):
            batch = enc[i:i+batch_size]
            batch_input = torch.tensor([seq[:-1] for seq in batch], dtype=torch.long).to(device)
            batch_target = torch.tensor([seq[1:] for seq in batch], dtype=torch.long).to(device)

            optimizer.zero_grad()

            # Forward pass returns logits and KL divergence
            logits, kl_divergence = model(batch_input)

            # Cross-entropy loss
            ce_loss = criterion(logits.view(-1, vocab_size), batch_target.view(-1))

            # Free-bits KL loss (Equation 5)
            # Only penalize KL divergence above threshold κ
            kl_per_token = kl_divergence.mean(dim=0)  # Average over batch, keep per-token
            kl_clamped = torch.clamp(kl_per_token - kappa, min=0.0)
            kl_loss = kl_clamped.mean()  # Average over sequence positions

            # Total loss
            loss = ce_loss + kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_kl_loss += kl_loss.item()

        avg_loss = total_loss / (len(enc) / batch_size)
        avg_ce_loss = total_ce_loss / (len(enc) / batch_size)
        avg_kl_loss = total_kl_loss / (len(enc) / batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f} (CE: {avg_ce_loss:.4f}, KL: {avg_kl_loss:.4f})")

        # Generate sample text
        model.eval()
        for _ in range(5):
            with torch.no_grad():
                import random
                letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                prompt = f"{letter}>"
                prompt_enc = [stoi.get(c, 0) for c in prompt]
                input_seq = torch.tensor([prompt_enc], dtype=torch.long).to(device)

                for _ in range(64):  # Generate exactly 64 characters
                    logits, _ = model(input_seq)  # Unpack tuple
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    input_seq = torch.cat([input_seq, next_token], dim=1)

                generated = ''.join([itos.get(i.item(), '?') for i in input_seq[0]])
                print(f"Generated: {generated}")
        model.train()

    # save the trained model
    torch.save(model.state_dict(), "models/free_transformer_model.pth")

if __name__ == "__main__":
    train()
