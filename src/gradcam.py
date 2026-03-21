import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_transforms
from model import DeepShieldModel

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def reshape_transform(tensor, height=14, width=14):
    """
    ViT outputs tokens not feature maps.
    This reshapes them into a spatial grid so GradCAM works.
    """
    # remove CLS token (first token)
    result = tensor[:, 1:, :]
    # reshape: (batch, 196, 768) → (batch, 14, 14, 768)
    result = result.reshape(result.size(0), height, width, result.size(2))
    # → (batch, 768, 14, 14)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run_gradcam(image_path, model_path="outputs/models/deepshield.pth"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── LOAD MODEL ────────────────────────────────────────────────
    model = DeepShieldModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # target the last transformer block's normalization layer
    # this is where ViT makes its final spatial decision
    target_layers = [model.backbone.blocks[-1].norm1]

    # ── LOAD IMAGE ────────────────────────────────────────────────
    transform = get_transforms("val")
    image_pil = Image.open(image_path).convert("RGB")

    # tensor for model input
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    # numpy array for visualization (0.0 to 1.0 range)
    image_np = np.array(image_pil.resize((224, 224))) / 255.0

    # ── RUN MODEL ─────────────────────────────────────────────────
    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)
        pred   = output.argmax(dim=1).item()
        confidence = probs[0][pred].item() * 100

    label = "FAKE" if pred == 1 else "REAL"
    print(f"\n🔍 Prediction: {label} ({confidence:.1f}% confidence)")

    # ── GRAD-CAM ──────────────────────────────────────────────────
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    # target class 1 = FAKE
    # shows what regions made the model say FAKE
    targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    # overlay heatmap on image
    visualization = show_cam_on_image(
        image_np.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    # ── PLOT ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(visualization)
    axes[2].set_title(f"Prediction: {label}\n{confidence:.1f}% confident", fontsize=13)
    axes[2].axis("off")

    plt.suptitle("DeepShield — Grad-CAM Explainability", fontsize=15, fontweight='bold')
    plt.tight_layout()

    os.makedirs("outputs/gradcam", exist_ok=True)
    save_path = "outputs/gradcam/result.png"
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Saved to {save_path}")


if __name__ == "__main__":
    # test on a fake image from our dataset
    test_image = "data/raw/Dataset/Test/Fake/fake_1.jpg"

    if not os.path.exists(test_image):
        # find any image in test folder
        for root, dirs, files in os.walk("data/raw/Dataset/Test"):
            for f in files:
                if f.endswith(('.jpg', '.png')):
                    test_image = os.path.join(root, f)
                    break
            break

    print(f"📸 Running Grad-CAM on: {test_image}")
    run_gradcam(test_image)