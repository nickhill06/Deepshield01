import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import DeepfakeDataset, get_transforms, build_dataframe
from model import DeepShieldModel


CONFIG = {
    "train_folder":  "data/raw/Dataset/Train",
    "val_folder":    "data/raw/Dataset/Validation",
    "batch_size":    32,
    "num_epochs":    15,
    "learning_rate": 1e-3,     # higher lr since only training head
    "save_path":     "outputs/models/deepshield.pth",
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "max_samples":   10000     # 5k real + 5k fake
}

print(f"🖥️  Using: {CONFIG['device'].upper()}")
if CONFIG['device'] == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


def train():

    # ── STEP 1: LOAD DATA ─────────────────────────────────────────
    print("\n📂 Loading dataset...")
    train_df = build_dataframe(CONFIG["train_folder"])
    val_df   = build_dataframe(CONFIG["val_folder"])

    max_per_class = CONFIG["max_samples"] // 2

    train_real = train_df[train_df["label"]==0].head(max_per_class)
    train_fake = train_df[train_df["label"]==1].head(max_per_class)
    train_df   = pd.concat([train_real, train_fake]).sample(frac=1).reset_index(drop=True)

    val_real = val_df[val_df["label"]==0].head(1000)
    val_fake = val_df[val_df["label"]==1].head(1000)
    val_df   = pd.concat([val_real, val_fake]).sample(frac=1).reset_index(drop=True)

    print(f"   Train: {len(train_df)} images")
    print(f"   Val:   {len(val_df)} images")

    train_dataset = DeepfakeDataset(train_df, transform=get_transforms("train"))
    val_dataset   = DeepfakeDataset(val_df,   transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
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

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")

    # ── STEP 2: BUILD MODEL ───────────────────────────────────────
    print("\n🧠 Loading model...")
    model = DeepShieldModel(pretrained=True).to(CONFIG["device"])

    # freeze ALL backbone parameters permanently
    # we only train the classifier head
    # WHY: backbone already knows faces from ImageNet
    # we just need to teach the head real vs fake
    for param in model.backbone.parameters():
        param.requires_grad = False

    # count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,}")
    print(f"   Backbone: FROZEN ❄️")

    # ── STEP 3: LOSS + OPTIMIZER ──────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # only pass classifier parameters to optimizer
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=CONFIG["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["num_epochs"]
    )

    # ── STEP 4: TRACKING ─────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    os.makedirs("outputs/models", exist_ok=True)

    # ── STEP 5: TRAINING LOOP ────────────────────────────────────
    print("\n🚀 Starting training...\n")

    for epoch in range(CONFIG["num_epochs"]):

        # ── TRAIN ────────────────────────────────────────────────
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        loop = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]"
        )

        for images, labels in loop:
            images = images.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # ── VALIDATION ───────────────────────────────────────────
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            loop = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Val]  "
            )

            for images, labels in loop:
                images = images.to(CONFIG["device"])
                labels = labels.to(CONFIG["device"])

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                loop.set_postfix(loss=f"{loss.item():.4f}")

        # ── SUMMARY ──────────────────────────────────────────────
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train — Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val   — Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

        # save best model based on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"   💾 Best model saved! Val Acc: {val_acc:.2f}%")

        scheduler.step()
        print()

    print(f"🏆 Best Val Accuracy: {best_val_acc:.2f}%")
    plot_history(history)
    print("✅ Training complete!")


def plot_history(history):
    os.makedirs("outputs/plots", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss", marker='o')
    ax1.plot(history["val_loss"],   label="Val Loss",   marker='o')
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train Acc", marker='o')
    ax2.plot(history["val_acc"],   label="Val Acc",   marker='o')
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy %")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/plots/training_curves.png")
    print("📈 Training curves saved!")


if __name__ == "__main__":
    train()