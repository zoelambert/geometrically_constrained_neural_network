import os

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from geometrically_constrained.conf.conf_file import config
from geometrically_constrained.train.loss import dice_loss, geometric_loss
from geometrically_constrained.train.primal_dual import splitting_algo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    dst_train: Dataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
) -> None:
    """
    Function to train the model.

    Args:
        dst_train (Dataset): Training dataset.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to be trained.
    """
    # Optimizer and learning rate scheduler setup
    learning_rate = config["train"]["learning_rate"]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=config["train"]["momentum"], weight_decay=config["train"]["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True)
    checkpoint_dir = config["train"]["checkpoint_dir"]

    # Geometry parameters
    mu = config["train"]["mu"]
    tau = config["train"]["tau"]
    sigma = config["train"]["sigma"]

    # Storage of variables u and w
    nb_samples = len(dst_train)
    M, N = dst_train[0][0].shape[1], dst_train[0][0].shape[2]  # Shape of 2D images MxN
    array = torch.zeros((nb_samples, 4, 2, M, N), device=DEVICE)

    for epoch in range(config["train"]["nb_epoch"]):
        train_losses = []
        model.train()

        for i, sample in enumerate(train_loader):
            images, masks, index = (
                sample[0].to(DEVICE),
                sample[1].long().to(DEVICE),
                sample[2],
            )

            optimizer.zero_grad()
            outputs = model(images)

            u, w = array[index, :, 0], array[index, :, 1]

            loss = geometric_loss(outputs, masks, u, w)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # Update u and w
            u_upgraded, w_upgraded = splitting_algo(
                outputs.detach(), masks, w, mu, tau, sigma
            )
            array[index, :, 0] = u_upgraded
            array[index, :, 1] = w_upgraded

            # Save model state
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss.item(),
            }
            torch.save(
                state,
                os.path.join(checkpoint_dir, f"sUNet_geometrical_epoch_{epoch}.pth"),
            )

        # Evaluation on validation data
        model.eval()
        val_losses = []

        # with torch.no_grad():
        for sample in val_loader:
            images, masks = (
                sample[0].to(DEVICE),
                sample[1].long().to(DEVICE),
            )

            outputs = model(images)
            loss = dice_loss(outputs, masks)
            val_losses.append(loss.item())

        # Learning rate scheduler step
        scheduler.step(np.mean(val_losses))

        # Print loss
        print(
            f"Epoch: {epoch}. Train Loss: {np.mean(train_losses):.{5}f}. Val Loss: {np.mean(val_losses):.{5}f}."
        )
