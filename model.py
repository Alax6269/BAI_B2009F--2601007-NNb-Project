"""model.py - the network definition and custom dataset grouping.

This includes FruitMLP and GroupedImageFolder to aggregate sub-categories 
(e.g., 'Apple Red 1') into broad classes (e.g., 'Apple').
"""
from torch import nn
from torchvision import datasets


class FruitMLP(nn.Module):
    def __init__(self, num_classes, img_size=100, hidden1=128, hidden2=64):
        super().__init__()
        input_size = img_size * img_size * 3   # 100 x 100 x 3 = 30,000 pixel values
        self.net = nn.Sequential(
            nn.Flatten(),                      # image -> one long vector
            nn.Linear(input_size, hidden1),    # hidden layer 1
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),       # hidden layer 2
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),   # output layer: one raw score per class
        )

    def forward(self, x):
        return self.net(x)


class GroupedImageFolder(datasets.ImageFolder):
    """Custom ImageFolder that groups sub-categories (e.g., 'Apple Red 1') 
    into broad classes (e.g., 'Apple') based on their base category name.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
        self.original_classes = self.classes
        new_classes = []
        
        for idx, sample in enumerate(self.samples):
            path, target_idx = sample
            orig_class_name = self.original_classes[target_idx]
            
            # Grouping rule: Take the first word as the broad category name
            broad_name = orig_class_name.split()[0].capitalize()
            
            if broad_name not in new_classes:
                new_classes.append(broad_name)
            
            new_target_idx = new_classes.index(broad_name)
            self.samples[idx] = (path, new_target_idx)
            
        self.targets = [s[1] for s in self.samples]
        self.classes = new_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(new_classes)}


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_architecture(model):
    """Print layer type, shape and parameter count."""
    print(f"{'Layer':<12}{'Details':<28}{'Parameters':>12}")
    print("-" * 52)
    for name, layer in model.net.named_children():
        params = sum(p.numel() for p in layer.parameters())
        if hasattr(layer, "in_features"):
            details = f"Linear {layer.in_features} -> {layer.out_features}"
        else:
            details = type(layer).__name__
        print(f"{name:<12}{details:<28}{params:>12,}")
    print("-" * 52)
    print(f"{'Total':<40}{count_parameters(model):>12,}")