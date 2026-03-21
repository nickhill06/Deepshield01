import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import DeepfakeDataset, get_transforms, build_dataframe
from model import DeepShieldModel


def evaluate(model_path="outputs/models/deepshield.pth"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using: {device.upper()}")

    # ── LOAD TEST DATA ────────────────────────────────────────────
    print("\n📂 Loading test data...")
    test_df = build_dataframe("data/raw/Dataset/Test")

    # use 2000 test images
    test_real = test_df[test_df["label"]==0].head(1000)
    test_fake = test_df[test_df["label"]==1].head(1000)
    test_df   = pd.concat([test_real, test_fake]).sample(frac=1).reset_index(drop=True)

    print(f"   Test images: {len(test_df)}")

    test_dataset = DeepfakeDataset(test_df, transform=get_transforms("val"))
    test_loader  = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # ── LOAD MODEL ────────────────────────────────────────────────
    print("\n🧠 Loading trained model...")
    model = DeepShieldModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("   Model loaded successfully!")

    # ── RUN PREDICTIONS ───────────────────────────────────────────
    print("\n🔍 Running predictions...")
    all_preds  = []   # predicted class (0 or 1)
    all_labels = []   # true class (0 or 1)
    all_probs  = []   # probability of being fake (for AUC-ROC)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)

            # softmax converts raw scores to probabilities
            probs = torch.softmax(outputs, dim=1)

            # argmax picks the class with highest probability
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of fake

    # ── METRICS ───────────────────────────────────────────────────
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)

    # classification report shows precision, recall, f1
    print("\n📋 Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Real", "Fake"]
    ))

    # AUC-ROC score — 1.0 is perfect, 0.5 is random
    auc = roc_auc_score(all_labels, all_probs)
    print(f"🎯 AUC-ROC Score: {auc:.4f}")

    # overall accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    print(f"✅ Overall Accuracy: {accuracy:.2f}%")

    # ── CONFUSION MATRIX ─────────────────────────────────────────
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        annot_kws={"size": 16}
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("Actual Label",    fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.show()
    print("   Saved to outputs/plots/confusion_matrix.png")

    # ── ROC CURVE ────────────────────────────────────────────────
    print("\n📈 Generating ROC curve...")
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate",  fontsize=12)
    plt.title("ROC Curve", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/plots/roc_curve.png")
    plt.show()
    print("   Saved to outputs/plots/roc_curve.png")

    print("\n✅ Evaluation complete!")
    print(f"   Check outputs/plots/ for graphs")


if __name__ == "__main__":
    evaluate()