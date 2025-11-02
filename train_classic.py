import torch

from model import TransformerModel
from utils import char_tokenizer
from dataset import generate_synthetic_data

def train():
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
    batch_size = 32
    num_epochs = 20
    lr = 1e-3

    # Initialize model, loss function, and optimizer
    model = TransformerModel(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers, dropout)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(enc), batch_size):
            batch = enc[i:i+batch_size]
            batch_input = torch.tensor([seq[:-1] for seq in batch], dtype=torch.long)
            batch_target = torch.tensor([seq[1:] for seq in batch], dtype=torch.long)

            optimizer.zero_grad()
            logits = model(batch_input)
            loss = criterion(logits.view(-1, vocab_size), batch_target.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(enc) / batch_size)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Generate sample text
        model.eval()
        for _ in range(5):
            with torch.no_grad():
                import random
                letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                prompt = f"{letter}>"
                prompt_enc = [stoi.get(c, 0) for c in prompt]
                input_seq = torch.tensor([prompt_enc], dtype=torch.long)

                for _ in range(64):  # Generate exactly 64 characters
                    logits = model(input_seq)
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    input_seq = torch.cat([input_seq, next_token], dim=1)

                generated = ''.join([itos.get(i.item(), '?') for i in input_seq[0]])
                print(f"Generated: {generated}")
        model.train()

    # save the trained model
    torch.save(model.state_dict(), "models/transformer_model.pth")

if __name__ == "__main__":
    train()
