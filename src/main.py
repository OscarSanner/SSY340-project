from colorization.dataloader import ColorDataset
from colorization.models import EnsembleHeadColorizer

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
import torch
import numpy as np
import logging
import datetime

logging.basicConfig(
    format="[%(asctime)s] - %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)


def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            input_imgs, true_img = x.to(device), y.to(device)
            pred = model.forward(input_imgs)

            batch_loss = loss_fn(pred, true_img)
            val_loss_cum += batch_loss.item()

    return val_loss_cum / len(val_loader)


def train_epoch(model, optimizer, loss_fn, train_loader, device):
    model.train()
    train_loss_batches = []

    for batch_index, (x, y) in enumerate(train_loader, 1):
        if batch_index % (len(train_loader) // 5) == 0:
            logging.info(f"Batch {batch_index}/{len(train_loader)}")
        # logging.info(f"Batch {batch_index}/{len(train_loader)}")

        input_imgs, true_img = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model.forward(input_imgs)
        loss = loss_fn(pred, true_img)
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

    return model, train_loss_batches


def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device
):
    logging.info("Starting training")

    model.to(device)
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        model, train_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)
        val_loss = validate(model, loss_fn, val_loader, device)

        logging.info(
            f"Epoch {epoch}/{num_epochs}: \t"
            f"Train loss: {sum(train_loss)/len(train_loss):.3f}, \t"
            f"Val. loss: {val_loss:.3f}, "
        )

        train_losses.extend(train_loss)
        val_losses.append(val_loss)

    return model, train_losses, val_losses


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ---------------------- Prepare data ----------------------
    path_to_dataset = "dataset"
    batch_size = 64
    n_workers = 8
    pin_memory = True
    shuffle_dataloader = True
    train_ratio, val_ratio = 0.8, 0.2

    logging.info(f"Loading data...")
    dataset = ColorDataset(path_to_dataset)
    train_dataset, val_dataset = random_split(dataset, [train_ratio, val_ratio])
    logging.info(f"\t train_ratio: {train_ratio} => {len(train_dataset)} images")
    logging.info(f"\t val_ratio: {val_ratio} => {len(val_dataset)} images")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_dataloader,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_dataloader,
    )

    # ---------------------- Model training ----------------------
    num_epochs = 5
    learning_rate = 1e-4
    weight_decay = 1e-5

    logging.info(f"Training model...")
    model = EnsembleHeadColorizer()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    model, train_losses, val_losses = training_loop(
        model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs, device
    )
    logging.info(f"Done! Saving model...")

    current_time = datetime.datetime.now().strftime("%Y-%b-%d_%Hh%Mm%Ss")
    file_name = f"model_{current_time}.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "model_stats": {
                "path_to_dataset": path_to_dataset,
                "batch_size": batch_size,
                "n_workers": n_workers,
                "pin_memory": pin_memory,
                "shuffle_dataloader": shuffle_dataloader,
                "train_ratio": train_ratio,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "loss_fn": loss_fn.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "model": str(model)
            },
        }, file_name
    )
    logging.info(f"Model saved as: {file_name}")

if __name__ == "__main__":
    run()

"""
# Assuming that you named your model "first_model"
torch.save({'model_state_dict': first_model.state_dict(),
            'train_losses': first_train_losses,
            'train_accs': first_train_accs,
            'val_losses': first_val_losses,
            'val_accs': first_val_accs,
            }, "./first_model.ckpt")

# Example of creating and initialising model with a previously saved state dict:
saved_first_model = FirstCnn(64) # fill-in the arguments if needed
checkpoint = torch.load("first_model.ckpt")
saved_first_model.load_state_dict(checkpoint['model_state_dict'])

"""
