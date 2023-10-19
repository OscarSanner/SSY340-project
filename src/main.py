from colorization.dataloader import ColorDataset
from colorization.models import EnsembleHeadColorizer
from loss_functions import PerceptualLoss

import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
import torch
import numpy as np
import logging
import datetime
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from piq import ssim, psnr


logging.basicConfig(
    format="[%(asctime)s] - %(message)s", datefmt="%H:%M:%S", level=logging.INFO
)

def validate(model, loss_fn, val_loader, device):
    val_loss_cum, val_psnr_cum, val_ssim_cum = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            input_imgs, true_img = x.to(device), y.to(device)
            pred = model.forward(input_imgs)

            batch_loss = loss_fn(pred, true_img)
            batch_psnr, batch_ssim = calc_psnr_and_ssid_for_batch(pred, true_img)

            val_loss_cum += batch_loss.item()
            val_psnr_cum += batch_psnr.item()
            val_ssim_cum += batch_ssim.item()

    return val_loss_cum / len(val_loader), val_psnr_cum / len(val_loader), val_ssim_cum / len(val_loader)

def calc_psnr_and_ssid_for_batch(pred_batch, true_batch):
    pred_batch_rgb = torch.Tensor(lab2rgb(pred_batch.detach().cpu().numpy().transpose(0, 2, 3, 1))).permute(0,3,1,2)
    true_batch_rgb = torch.Tensor(lab2rgb(true_batch.detach().cpu().numpy().transpose(0, 2, 3, 1))).permute(0,3,1,2)
    return psnr(pred_batch_rgb, true_batch_rgb, reduction='mean', data_range=255), ssim(pred_batch_rgb, true_batch_rgb, reduction='mean', data_range=255)

def train_epoch(model, optimizer, loss_fn, train_loader, device):
    model.train()
    train_loss_batches, train_psnr_batches, train_ssim_batches = [], [], []

    for batch_index, (x, y) in enumerate(train_loader, 1):
        if batch_index % (len(train_loader) // 5) == 0:
           logging.info(f"Batch {batch_index}/{len(train_loader)}")
        #logging.info(f"Batch {batch_index}/{len(train_loader)}")

        input_imgs, true_img = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model.forward(input_imgs)

        loss = loss_fn(pred, true_img)
        batch_psnr, batch_ssim = calc_psnr_and_ssid_for_batch(pred, true_img)

        loss.backward()
        optimizer.step()

        train_loss_batches.append(loss.item())
        train_psnr_batches.append(batch_psnr.item())
        train_ssim_batches.append(batch_ssim.item())

    return model, train_loss_batches, train_psnr_batches, train_ssim_batches


def training_loop(
    model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device
):
    logging.info("Starting training")

    model.to(device)
    train_losses, val_losses, train_psnr_res, val_psnr_res, train_ssim_res, val_ssim_res = [], [], [], [], [], [] 

    for epoch in range(1, num_epochs + 1):
        model, train_loss, train_psnr, train_ssim = train_epoch(model, optimizer, loss_fn, train_loader, device)
        val_loss, val_psnr, val_ssim = validate(model, loss_fn, val_loader, device)

        logging.info(
            f"Epoch {epoch}/{num_epochs}: \t"
            f"Train loss: {sum(train_loss)/len(train_loss):.3f}, \t"
            f"Val. loss: {val_loss:.3f}, "
            f"Train PSNR: {sum(train_psnr)/len(train_psnr):.3f}, \t"
            f"Val. PSNR: {val_psnr:.3f}, "
            f"Train SSIM: {sum(train_ssim)/len(train_ssim):.3f}, \t"
            f"Val. SSIM: {val_ssim:.3f}, "
        )

        train_losses.extend(train_loss)
        val_losses.append(val_loss)

        train_psnr_res.extend(train_psnr)
        val_psnr_res.append(val_psnr)

        train_ssim_res.extend(train_ssim)
        val_ssim_res.append(val_ssim)

    return model, train_losses, val_losses, train_psnr_res, val_psnr_res, train_ssim_res, val_ssim_res

def plot_stats(train_losses, val_losses, train_psnr, val_psnr, train_ssim, val_ssim):

    # Calculate x values for plotting
    x_train = list(range(len(train_losses)))
    x_val = list(range(len(val_losses)))

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Losses plot
    axes[0].plot(x_train, train_losses, '-o', label='Train Losses')
    axes[0].plot(x_val, val_losses, '-o', label='Validation Losses')
    axes[0].set_title('Losses vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # PSNR plot
    axes[1].plot(x_train, train_psnr, '-o', label='Train PSNR')
    axes[1].plot(x_val, val_psnr, '-o', label='Validation PSNR')
    axes[1].set_title('PSNR vs. Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].legend()

    # SSIM plot
    axes[2].plot(x_train, train_ssim, '-o', label='Train SSIM')
    axes[2].plot(x_val, val_ssim, '-o', label='Validation SSIM')
    axes[2].set_title('SSIM vs. Epochs')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('SSIM')
    axes[2].legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("PLOTS")


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ---------------------- Prepare data ----------------------
    path_to_dataset = "dataset"
    batch_size = 16
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
    num_epochs = 4
    learning_rate = 1e-4
    weight_decay = 1e-5

    logging.info(f"Training model...")
    model = EnsembleHeadColorizer()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = PerceptualLoss(device)
 
    model, train_losses, val_losses, train_psnr, val_psnr, train_ssim, val_ssim = training_loop(
        model, optimizer, loss_fn, train_dataloader, val_dataloader, num_epochs, device
    )
    logging.info(f"Done! Saving model...")
    print(val_psnr)
    print(val_ssim)

    current_time = datetime.datetime.now().strftime("%Y-%b-%d_%Hh%Mm%Ss")
    file_name = f"model_{current_time}.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_psnr": train_psnr,
            "val_psnr": val_psnr,
            "train_ssim": train_ssim,
            "val_ssim": val_ssim,
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
    logging.info(f"Done! Saving model: {file_name}")
    plot_stats(train_losses, val_losses, train_psnr, val_psnr, train_ssim, val_ssim)

if __name__ == "__main__":
    run()
