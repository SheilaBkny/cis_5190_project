import argparse

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import Model
from preprocess import prepare_data


def train(csv_path: str, output_path: str = "model.pt", epochs: int = 5, batch_size: int = 32, lr: float = 1e-4) -> None:
    X, y = prepare_data(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    indices = list(range(len(X)))
    if len(indices) > 1:
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    else:
        train_idx, val_idx = indices, indices

    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[val_idx], y[val_idx]), batch_size=batch_size)

    model = Model(weights_path=None).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                val_loss += criterion(model(images), targets).item() * images.size(0)

        train_loss /= max(len(train_idx), 1)
        val_loss /= max(len(val_idx), 1)
        print(f"epoch {epoch}: train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    torch.save(model.state_dict(), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline Img2GPS model.")
    parser.add_argument("--csv", default="metadata.csv", help="Path to the CSV file containing image paths and GPS coordinates.")
    parser.add_argument("--output", default="model.pt", help="Where to save the model state_dict.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args.csv, args.output, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
