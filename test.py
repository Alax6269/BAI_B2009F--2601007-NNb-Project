# ==========================================
# Fruit Classification - Testing Script
# ==========================================

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import FruitMLP, GroupedImageFolder


def compute_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
        
    true_positives = np.diag(cm).astype(float)
    predicted_totals = cm.sum(axis=0)
    true_totals = cm.sum(axis=1)
    
    precision = np.divide(true_positives, predicted_totals, out=np.zeros(n_classes), where=predicted_totals > 0)
    recall = np.divide(true_positives, true_totals, out=np.zeros(n_classes), where=true_totals > 0)
    
    denominator = precision + recall
    f1_scores = np.divide(2 * precision * recall, denominator, out=np.zeros(n_classes), where=denominator > 0)
    
    overall_accuracy = true_positives.sum() / cm.sum()
    return cm, overall_accuracy, precision, recall, f1_scores, true_totals


def save_visualizations(cm, class_names, wrong_predictions, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("Matplotlib not found, skipping figure generation.")
        return

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n_classes * 0.8), max(4, n_classes * 0.7)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Test Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    top_mistakes = wrong_predictions[:10]
    if top_mistakes:
        cols = min(5, len(top_mistakes))
        rows = (len(top_mistakes) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.6), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, mistake in zip(axes.flat, top_mistakes):
            ax.imshow(Image.open(mistake["path"]).convert("RGB"))
            ax.set_title(f"True: {mistake['true']}\nPred: {mistake['pred']} ({mistake['confidence']:.0%})", fontsize=7)
        fig.suptitle("Most Confident Wrong Predictions", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / "wrong_predictions.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fruit Classification Model")
    parser.add_argument("--test-dir", default="data/Test", help="Path to test image folder")
    parser.add_argument("--model", default="models/baseline.pt", help="Path to saved model checkpoint")
    parser.add_argument("--classes", default="models/classes.json", help="Path to classes JSON file")
    parser.add_argument("--out", default="results", help="Where to save test results")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--require-gpu", action="store_true", help="Error out if no GPU is found")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise SystemExit("Error: --require-gpu flag set, but no CUDA GPU detected!")
    print(f"Evaluating using device: {device}")

    torch.manual_seed(42)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    
    model = FruitMLP(config["num_classes"], config["img_size"], config["hidden1"], config["hidden2"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    class_names = json.loads(Path(args.classes).read_text())
    
    transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
    ])
    test_dataset = GroupedImageFolder(args.test_dir, transform=transform)
    
    unknown_classes = [name for name in test_dataset.classes if name not in class_names]
    if unknown_classes:
        raise SystemExit(f"Error: Test folder contains broad classes the model was not trained on: {unknown_classes}")

    class_remap = [class_names.index(name) for name in test_dataset.classes]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    y_true, y_pred, confidences = [], [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            probabilities = torch.softmax(model(inputs), dim=1)
            conf, preds = probabilities.max(dim=1)
            
            y_pred.extend(preds.tolist())
            confidences.extend(conf.tolist())
            y_true.extend([class_remap[label.item()] for label in labels])

    n_classes = len(class_names)
    cm, accuracy, precision, recall, f1, support = compute_metrics(y_true, y_pred, n_classes)
    valid_classes = support > 0

    print(f"\nEvaluation Results:")
    print(f"Total Test Images: {len(y_true)} | Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision[valid_classes].mean():.4f} | "
          f"Macro Recall: {recall[valid_classes].mean():.4f} | "
          f"Macro F1: {f1[valid_classes].mean():.4f}")

    wrong_predictions = [
        {
            "path": test_dataset.samples[i][0],
            "true": class_names[y_true[i]],
            "pred": class_names[y_pred[i]],
            "confidence": confidences[i]
        }
        for i in range(len(y_true)) if y_true[i] != y_pred[i]
    ]
    wrong_predictions.sort(key=lambda item: -item["confidence"])

    with open(output_dir / "wrong_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true", "pred", "confidence"])
        writer.writeheader()
        writer.writerows(wrong_predictions)

    metrics_summary = {
        "test_images": len(y_true),
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(precision[valid_classes].mean()), 4),
        "macro_recall": round(float(recall[valid_classes].mean()), 4),
        "macro_f1": round(float(f1[valid_classes].mean()), 4),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics_summary, indent=2))
    
    save_visualizations(cm, class_names, wrong_predictions, output_dir)
    print(f"Saved all test artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()