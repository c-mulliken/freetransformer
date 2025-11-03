"""
Evaluation script for analyzing latent structure discovery in Free Transformer.

Tests whether the model learns meaningful latent representations without supervision,
specifically examining what information is encoded in the latent variable Z.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
import seaborn as sns

from model import FreeTransformer
from utils import char_tokenizer
from dataset import generate_synthetic_data


def extract_latent_codes(model, data, stoi, device, num_samples=1000):
    """
    Extract latent codes Z from the encoder for a dataset.

    Returns:
        latent_indices: (num_samples, seq_len) - integer indices of latent codes
        metadata: dict with ground truth information about each sample
    """
    model.eval()

    latent_indices = []
    metadata = {
        'target_letters': [],
        'target_positions': [],
        'sequences': [],
        'prompts': [],
    }

    with torch.no_grad():
        for i, seq_str in enumerate(data[:num_samples]):
            # Parse the sequence
            # Format: "LETTER>____TARGET____..." where TARGET is 8 repeated letters
            prompt = seq_str[:2]  # e.g., "A>"
            sequence = seq_str[2:]  # The 64-character sequence

            # Find target letter and position
            target_letter = prompt[0]

            # Find position of target (8 consecutive same letters)
            target_pos = -1
            for pos in range(len(sequence) - 7):
                if len(set(sequence[pos:pos+8])) == 1 and sequence[pos] == target_letter:
                    target_pos = pos
                    break

            # Tokenize
            seq_enc = [stoi.get(c, 0) for c in seq_str]
            input_seq = torch.tensor([seq_enc[:-1]], dtype=torch.long).to(device)

            # Forward pass through first half + encoder
            x = model.token_embedding(input_seq)
            x = model.positional_encoding(x)

            half_layers = len(model.layers) // 2
            for layer in model.layers[:half_layers]:
                x = layer(x)

            # Get encoder output
            enc_logits = model.ft_encoder(x)  # (1, T, H)

            # Get latent codes (using deterministic mapping)
            p = torch.sigmoid(enc_logits)
            bits = (p > 0.5).float()
            powers = model.binary_mapper.powers
            indices = (bits * powers).sum(dim=-1).long()  # (1, T)

            latent_indices.append(indices.cpu().numpy()[0])
            metadata['target_letters'].append(target_letter)
            metadata['target_positions'].append(target_pos)
            metadata['sequences'].append(sequence)
            metadata['prompts'].append(prompt)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")

    latent_indices = np.array(latent_indices)
    return latent_indices, metadata


def analyze_latent_structure(latent_indices, metadata, output_dir="eval_results"):
    """
    Analyze what information is encoded in the latent variables.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    num_samples, seq_len = latent_indices.shape

    # 1. Check if latent codes vary by position in sequence
    print("\n=== Analysis 1: Latent Code Variation ===")
    unique_codes_per_position = [len(np.unique(latent_indices[:, t])) for t in range(seq_len)]
    print(f"Unique codes per position (first 10): {unique_codes_per_position[:10]}")
    print(f"Unique codes per position (last 10): {unique_codes_per_position[-10:]}")

    plt.figure(figsize=(12, 4))
    plt.plot(unique_codes_per_position)
    plt.xlabel("Position in Sequence")
    plt.ylabel("Number of Unique Latent Codes")
    plt.title("Latent Code Diversity Across Sequence Positions")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/latent_diversity_by_position.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Check correlation between latent code and target letter
    print("\n=== Analysis 2: Latent Code vs Target Letter ===")

    # For each position, compute how well latent code predicts target letter
    letter_prediction_accuracy = []
    for t in range(seq_len):
        codes = latent_indices[:, t]
        letters = metadata['target_letters']

        # Most common letter for each latent code
        code_to_letter = defaultdict(lambda: defaultdict(int))
        for code, letter in zip(codes, letters):
            code_to_letter[code][letter] += 1

        # Accuracy if we predict most common letter for each code
        correct = 0
        for code, letter in zip(codes, letters):
            most_common = max(code_to_letter[code].items(), key=lambda x: x[1])[0]
            if most_common == letter:
                correct += 1

        letter_prediction_accuracy.append(correct / num_samples)

    plt.figure(figsize=(12, 4))
    plt.plot(letter_prediction_accuracy)
    plt.xlabel("Position in Sequence")
    plt.ylabel("Target Letter Prediction Accuracy")
    plt.title("How Well Does Latent Code Predict Target Letter?")
    plt.axhline(y=1/26, color='r', linestyle='--', label='Random Baseline (1/26)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/letter_prediction_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Letter prediction accuracy at position 0: {letter_prediction_accuracy[0]:.3f}")
    print(f"Letter prediction accuracy at position 32: {letter_prediction_accuracy[32]:.3f}")

    # 3. Check correlation between latent code and target position
    print("\n=== Analysis 3: Latent Code vs Target Position ===")

    position_prediction_mae = []
    for t in range(seq_len):
        codes = latent_indices[:, t]
        positions = metadata['target_positions']

        # Average position for each latent code
        code_to_positions = defaultdict(list)
        for code, pos in zip(codes, positions):
            if pos >= 0:  # Valid position found
                code_to_positions[code].append(pos)

        code_to_avg_pos = {code: np.mean(positions) for code, positions in code_to_positions.items()}

        # MAE if we predict average position for each code
        errors = []
        for code, pos in zip(codes, positions):
            if pos >= 0 and code in code_to_avg_pos:
                predicted = code_to_avg_pos[code]
                errors.append(abs(predicted - pos))

        if errors:
            position_prediction_mae.append(np.mean(errors))
        else:
            position_prediction_mae.append(64)  # Max error

    plt.figure(figsize=(12, 4))
    plt.plot(position_prediction_mae)
    plt.xlabel("Position in Sequence")
    plt.ylabel("Target Position Prediction MAE")
    plt.title("How Well Does Latent Code Predict Target Position?")
    plt.axhline(y=64/2, color='r', linestyle='--', label='Random Baseline (32)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/position_prediction_mae.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Position prediction MAE at position 0: {position_prediction_mae[0]:.1f}")
    print(f"Position prediction MAE at position 32: {position_prediction_mae[32]:.1f}")

    # 4. Visualize latent codes with t-SNE (at a specific position)
    print("\n=== Analysis 4: t-SNE Visualization ===")

    # Choose middle position for visualization
    vis_position = seq_len // 2
    codes_at_pos = latent_indices[:, vis_position]

    # Create embeddings by treating each unique code as a one-hot
    unique_codes = np.unique(codes_at_pos)
    print(f"Number of unique codes at position {vis_position}: {len(unique_codes)}")

    if len(unique_codes) > 2:  # Need at least 3 points for t-SNE
        # Map codes to indices
        code_to_idx = {code: idx for idx, code in enumerate(unique_codes)}
        code_indices = np.array([code_to_idx[code] for code in codes_at_pos])

        # One-hot encode
        one_hot = np.eye(len(unique_codes))[code_indices]

        # Run t-SNE
        if len(unique_codes) > 50:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(unique_codes) - 1))
            embeddings = tsne.fit_transform(one_hot)
        else:
            pca = PCA(n_components=2)
            embeddings = pca.fit_transform(one_hot)

        # Plot colored by target letter
        plt.figure(figsize=(10, 8))
        letters = metadata['target_letters']
        unique_letters = sorted(set(letters))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_letters)))

        for letter, color in zip(unique_letters, colors):
            mask = [l == letter for l in letters]
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       label=letter, alpha=0.6, s=50, color=color)

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title(f"Latent Code Embeddings at Position {vis_position} (Colored by Target Letter)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tsne_by_letter.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot colored by target position
        plt.figure(figsize=(10, 8))
        positions = np.array(metadata['target_positions'])
        valid_mask = positions >= 0

        scatter = plt.scatter(embeddings[valid_mask, 0], embeddings[valid_mask, 1],
                            c=positions[valid_mask], cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Target Position')
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title(f"Latent Code Embeddings at Position {vis_position} (Colored by Target Position)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tsne_by_position.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Check if latent codes are consistent across sequence
    print("\n=== Analysis 5: Temporal Consistency ===")

    # For each sample, how many unique codes are used?
    codes_per_sample = [len(np.unique(latent_indices[i])) for i in range(num_samples)]

    plt.figure(figsize=(10, 6))
    plt.hist(codes_per_sample, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel("Number of Unique Codes per Sample")
    plt.ylabel("Count")
    plt.title("Distribution of Unique Latent Codes Used per Sample")
    plt.axvline(x=np.mean(codes_per_sample), color='r', linestyle='--',
                label=f'Mean: {np.mean(codes_per_sample):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/codes_per_sample.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Average unique codes per sample: {np.mean(codes_per_sample):.1f}")
    print(f"If model uses 1 code per sample: 1.0")
    print(f"If model uses different code per position: ~{seq_len}")

    print(f"\nAll visualizations saved to {output_dir}/")


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate test data
    print("Generating test data...")
    test_data = generate_synthetic_data(1000)
    _, stoi, _ = char_tokenizer(test_data)

    # Load model
    print("Loading model...")
    vocab_size = len(stoi)
    max_len = 64 + 2
    embed_dim = 128
    num_heads = 4
    ff_dim = 512
    num_layers = 4
    H = 16
    dropout = 0.1

    model = FreeTransformer(
        vocab_size, max_len, embed_dim, num_heads,
        ff_dim, num_layers, H, dropout
    ).to(device)

    # Load trained weights
    model_path = "models/free_transformer_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Model file {model_path} not found. Using randomly initialized model.")
        print("Results will not be meaningful without a trained model!")

    # Extract latent codes
    print("\nExtracting latent codes...")
    latent_indices, metadata = extract_latent_codes(model, test_data, stoi, device)

    # Analyze
    print("\nAnalyzing latent structure...")
    analyze_latent_structure(latent_indices, metadata)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
