import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from geometrically_constrained.dataset.dataset import get_data
from geometrically_constrained.model.model import get_model
from geometrically_constrained.train.train import train


def main() -> None:
    """
    Main function to train the geometrically constrained model.
    """
    # Load data
    dst_train: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    dst_train, train_loader, val_loader = get_data()

    # Load model
    model: nn.Module
    model = get_model()

    # Train model
    train(dst_train, train_loader, val_loader, model)


if __name__ == "__main__":
    main()
