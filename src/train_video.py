import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from model import DeepShieldModel

# ── CONFIG ─────────────────────────────────────────────────────
CONFIG = {
    "data_dir"   : "data/video_faces",
    "model_save" : "outputs/models/deepshield.pth",
    "batch_size" : 16,
    "num_epochs" : 20,
    "lr"         : 1e-3,
    "val_split"  : 0.2,
    "device"     : "cuda" if torch.cuda.is_available() else "cpu"
}

# ── TRANSFORMS ─────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ── DATASET ────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_dataset(data_dir):
    """Load all images from real/ and fake/ folders"""
    image_paths = []
    labels      = []

    # real = 0, fake = 1
    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"WARNING: {folder_path} not found!")
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(folder_path, fname))
                labels.append(label)

    return image_paths, labels


# ── TRAINING ───────────────────────────────────────────────────
def train():
    device = torch.device(CONFIG["device"])
    print(f"\n{'='*50}")
    print("   DEEPSHIELD — Retraining on Video Faces")
    print(f"{'='*50}")
    print(f"   Device: {device}")
    if device.type == "cuda":
        print(f"   GPU:    {torch.cuda.get_device_name(0)}")

    # ── LOAD DATA ──────────────────────────────────────────────
    print("\n   Loading dataset...")
    image_paths, labels = build_dataset(CONFIG["data_dir"])

    real_count = labels.count(0)
    fake_count = labels.count(1)
    print(f"   Real images: {real_count}")
    print(f"   Fake images: {fake_count}")
    print(f"   Total:       {len(labels)}")

    # ── TRAIN/VAL SPLIT ────────────────────────────────────────
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=CONFIG["val_split"],
        random_state=42,
        stratify=labels   # keep balance in both splits
    )

    print(f"\n   Train: {len(train_paths)} images")
    print(f"   Val:   {len(val_paths)} images")

    # ── WEIGHTED SAMPLER (fix imbalance) ───────────────────────
    class_counts  = [train_labels.count(0), train_labels.count(1)]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # ── DATALOADERS ────────────────────────────────────────────
    train_dataset = FaceDataset(train_paths, train_labels, train_transform)
    val_dataset   = FaceDataset(val_paths,   val_labels,   val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ── MODEL ──────────────────────────────────────────────────
    print("\n   Loading model...")
    model = DeepShieldModel()

    # load existing weights
    checkpoint = torch.load(CONFIG["model_save"], map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("   Loaded existing model weights.")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        print("   Loaded existing model weights.")
    else:
        model.load_state_dict(checkpoint)
        print("   Loaded existing model weights.")

    model.to(device)

    # freeze backbone, only train classifier
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("   Backbone frozen. Training classifier only.")

    # ── OPTIMIZER & LOSS ───────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=CONFIG["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    # ── TRAINING LOOP ──────────────────────────────────────────
    best_val_acc  = 0.0
    train_accs    = []
    val_accs      = []
    train_losses  = []
    val_losses    = []

    print(f"\n   Starting training for {CONFIG['num_epochs']} epochs...")
    print(f"{'='*50}")

    for epoch in range(CONFIG["num_epochs"]):
        # TRAIN
        model.train()
        correct = 0
        total   = 0
        running_loss = 0.0

        for images, labels_batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]",
            leave=False
        ):
            images       = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            total        += labels_batch.size(0)
            correct      += predicted.eq(labels_batch).sum().item()

        train_acc  = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # VALIDATE
        model.eval()
        val_correct = 0
        val_total   = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images       = images.to(device)
                labels_batch = labels_batch.to(device)
                outputs      = model(images)
                loss         = criterion(outputs, labels_batch)
                val_loss_sum += loss.item()
                _, predicted  = outputs.max(1)
                val_total    += labels_batch.size(0)
                val_correct  += predicted.eq(labels_batch).sum().item()

        val_acc  = 100.0 * val_correct / val_total
        val_loss = val_loss_sum / len(val_loader)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"   Epoch {epoch+1:02d}/{CONFIG['num_epochs']} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | "
              f"Loss: {val_loss:.4f}")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch
            }, CONFIG["model_save"])
            print(f"   ✅ Best model saved! Val acc: {val_acc:.1f}%")

    print(f"\n{'='*50}")
    print(f"   Training complete!")
    print(f"   Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"{'='*50}")

    # ── PLOT ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0f")

    ax1.plot(train_accs, color="#00d4ff", label="Train Acc")
    ax1.plot(val_accs,   color="#a78bfa", label="Val Acc")
    ax1.set_facecolor("#0f172a")
    ax1.set_title("Accuracy", color="white")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Accuracy %", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#0f172a", labelcolor="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#334155")

    ax2.plot(train_losses, color="#00d4ff", label="Train Loss")
    ax2.plot(val_losses,   color="#a78bfa", label="Val Loss")
    ax2.set_facecolor("#0f172a")
    ax2.set_title("Loss", color="white")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Loss", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#0f172a", labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#334155")

    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/training_curves_v2.png",
                dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()
    plt.close()


if __name__ == "__main__":
    train()