# ==========================================
# Fruit Classification - Training Script
# ==========================================

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from model import FruitMLP, GroupedImageFolder, count_parameters, print_architecture


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(targets, val_fraction, seed):
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    train_idx, val_idx = [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()
    return train_idx, val_idx


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    
    with torch.set_grad_enabled(is_training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item() * len(labels)
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += len(labels)
            
    return total_loss / total_samples, correct_predictions / total_samples


def plot_curves(history, save_path, exp_name):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.legend()
    
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.legend()
    
    fig.suptitle(f"Experiment: {exp_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Fruit Classification MLP")
    parser.add_argument("--data-dir", default="data/Training", help="Path to training images")
    parser.add_argument("--model-dir", default="models", help="Where to save model weights")
    parser.add_argument("--results-dir", default="results", help="Where to save logs and plots")
    parser.add_argument("--name", default="baseline", help="Experiment name")
    parser.add_argument("--note", default="", help="Notes on what changed in this run")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden1", type=int, default=128, help="Size of first hidden layer")
    parser.add_argument("--hidden2", type=int, default=64, help="Size of second hidden layer")
    parser.add_argument("--img-size", type=int, default=100, help="Image resize dimension")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--require-gpu", action="store_true", help="Error out if no GPU is found")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise SystemExit("Error: --require-gpu flag was set, but no CUDA GPU was detected!")
    print(f"Using device: {device}")

    set_seed(args.seed)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # Use GroupedImageFolder to load and aggregate broad classes
    full_dataset = GroupedImageFolder(args.data_dir, transform=transform)
    class_names = full_dataset.classes
    
    train_idx, val_idx = stratified_split(full_dataset.targets, args.val_fraction, args.seed)
    print(f"Loaded {len(class_names)} broad classes: {class_names}")
    print(f"Training images: {len(train_idx)}, Validation images: {len(val_idx)}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=args.batch_size, 
                              shuffle=True, generator=generator, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=args.batch_size, 
                            shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    model = FruitMLP(len(class_names), args.img_size, args.hidden1, args.hidden2).to(device)
    print_architecture(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_model_state = {"val_loss": float("inf"), "val_acc": 0.0, "epoch": 0}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

        if val_loss < best_model_state["val_loss"]:
            best_model_state = {"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch}
            torch.save({
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": {
                    "num_classes": len(class_names),
                    "img_size": args.img_size,
                    "hidden1": args.hidden1,
                    "hidden2": args.hidden2
                },
            }, model_dir / f"{args.name}.pt")

    (model_dir / "classes.json").write_text(json.dumps(class_names, indent=2))
    plot_curves(history, results_dir / f"{args.name}_curves.png", args.name)
    (results_dir / f"{args.name}_history.json").write_text(json.dumps(history, indent=2))

    log_file = results_dir / "experiment_log.csv"
    file_is_new = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if file_is_new:
            writer.writerow(["exp", "what_changed", "epochs", "lr", "batch", "hidden1", "hidden2", 
                             "params", "best_epoch", "val_loss", "val_acc"])
        writer.writerow([args.name, args.note, args.epochs, args.lr, args.batch_size, 
                         args.hidden1, args.hidden2, count_parameters(model), 
                         best_model_state["epoch"], f"{best_model_state['val_loss']:.4f}", 
                         f"{best_model_state['val_acc']:.4f}"])

    print(f"\nTraining complete! Best Epoch #{best_model_state['epoch']} "
          f"Val Loss: {best_model_state['val_loss']:.4f} | Val Acc: {best_model_state['val_acc']:.3f}")


if __name__ == "__main__":
    main()