r"""predict.py - classify the specified image with the saved model."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import FruitMLP


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default="Image For Model To Predict/image 1.jpg", help="path to the image file")
    ap.add_argument("--model", default="models/baseline.pt")
    ap.add_argument("--classes", default="models/classes.json")
    ap.add_argument("--top", type=int, default=3, help="how many top guesses to show")
    classes_path = Path("models/classes.json")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Error: Image path '{args.image}' not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = FruitMLP(cfg["num_classes"], cfg["img_size"], cfg["hidden1"], cfg["hidden2"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    classes = json.loads(Path(args.classes).read_text())

    tfm = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ])
    image = Image.open(args.image).convert("RGB")
    x = tfm(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x.to(device)), dim=1)[0]
    conf, idx = probs.topk(min(args.top, len(classes)))
    conf, idx = conf.tolist(), idx.tolist()

    print(f"Image: {args.image}")
    print(f"Prediction: {classes[idx[0]]}  ({conf[0]:.1%} confidence)")
    if len(idx) > 1:
        print("Top guesses:")
        for c, i in zip(conf, idx):
            print(f"   {classes[i]:<28}{c:.1%}")


if __name__ == "__main__":
    main()