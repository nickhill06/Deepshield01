import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(os.path.dirname(__file__))
from model import DeepShieldModel

# ── CONFIG ─────────────────────────────────────────────────────
THRESHOLD    = 5.0    # % of fake frames to call FAKE
AVG_PROB_THR = 0.05   # average fake probability threshold
MODEL_PATH   = "outputs/models/deepshield.pth"
SAVE_DIR     = "outputs/gradcam"
FPS_SAMPLE   = 5
FACE_CASCADE = "src/haarcascade_frontalface_default.xml"

# ── TRANSFORMS ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_model(model_path, device):
    model = DeepShieldModel()
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    return result.transpose(2, 3).transpose(1, 2)


def detect_and_crop_face(frame_rgb, face_cascade, padding=40):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None, None
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest
    h_img, w_img = frame_rgb.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    face_crop = frame_rgb[y1:y2, x1:x2]
    return face_crop, (x1, y1, x2, y2)


def run_gradcam(model, frame_rgb, device, save_path):
    target_layers = [model.backbone.blocks[-1].norm1]
    img_tensor = transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    img_float   = frame_rgb.astype(np.float32) / 255.0
    img_resized = cv2.resize(img_float, (224, 224))
    visualization = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0f")
    axes[0].imshow(img_resized)
    axes[0].set_title("Face Crop", color="white", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", color="white", fontsize=12)
    axes[1].axis("off")
    axes[2].imshow(visualization)
    axes[2].set_title("Overlay", color="white", fontsize=12)
    axes[2].axis("off")
    for ax in axes:
        ax.set_facecolor("#0a0a0f")
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "result.png"),
                dpi=150, bbox_inches="tight",
                facecolor="#0a0a0f")
    plt.show()
    plt.close()


def predict_video(video_path, model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n   Using device: {device}")

    print("   Loading model...")
    model = load_model(model_path, device)

    if not os.path.exists(FACE_CASCADE):
        print(f"   WARNING: Face cascade not found at {FACE_CASCADE}")
        print("   Running without face detection...")
        face_cascade = None
    else:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
        print("   Face detector loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    step     = max(1, int(fps / FPS_SAMPLE))

    print(f"   Duration:     {duration:.1f} seconds")
    print(f"   FPS:          {fps}")
    print(f"   Resolution:   {width}x{height}")
    print(f"   Total frames: {total}")
    print(f"   Will analyze ~{total // step} frames (every {step}th frame)")

    fake_probs       = []
    frame_indices    = []
    suspicious_frame = None
    suspicious_face  = None
    max_fake_prob    = 0.0
    frame_idx        = 0
    analyzed         = 0
    faces_found      = 0
    no_face_count    = 0

    print("\n   Analyzing frames with face detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if face_cascade is not None:
                face_crop, bbox = detect_and_crop_face(frame_rgb, face_cascade)
            else:
                face_crop = frame_rgb
                bbox      = None

            if face_crop is None:
                no_face_count += 1
                frame_idx += 1
                continue

            faces_found += 1
            pil_img = Image.fromarray(face_crop)
            tensor  = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob   = torch.softmax(output, dim=1)[0][1].item()

            fake_probs.append(prob)
            frame_indices.append(frame_idx)
            analyzed += 1

            if prob > max_fake_prob:
                max_fake_prob    = prob
                suspicious_frame = frame_rgb.copy()
                suspicious_face  = face_crop.copy()

            if analyzed % 10 == 0:
                print(f"   Analyzed {analyzed} frames ({faces_found} faces found)...")

        frame_idx += 1

    cap.release()

    print(f"\n   Total frames analyzed: {analyzed}")
    print(f"   Faces detected:        {faces_found}")
    print(f"   Frames skipped:        {no_face_count}")

    if analyzed == 0:
        print("\n   WARNING: No faces detected in video!")
        return "UNKNOWN", 0.0

    fake_count    = sum(1 for p in fake_probs if p > 0.5)
    total_frames  = len(fake_probs)
    fake_percent  = (fake_count / total_frames) * 100
    avg_fake_prob = np.mean(fake_probs) * 100

    verdict = "FAKE" if (fake_percent > THRESHOLD or
                         avg_fake_prob > AVG_PROB_THR * 100) else "REAL"

    print("\n" + "=" * 50)
    print("   VIDEO ANALYSIS RESULTS")
    print("=" * 50)
    print(f"   File:             {os.path.basename(video_path)}")
    print(f"   Frames with face: {total_frames}")
    print(f"   Fake frames:      {fake_count} ({fake_percent:.1f}%)")
    print(f"   Real frames:      {total_frames - fake_count} ({100-fake_percent:.1f}%)")
    print(f"   Avg fake prob:    {avg_fake_prob:.1f}%")
    print(f"   Max fake prob:    {max_fake_prob*100:.1f}%")
    print(f"   Threshold:        >{THRESHOLD}% frames OR avg >{AVG_PROB_THR*100}%")
    print(f"\n   VERDICT: {verdict} VIDEO")
    print("=" * 50)

    if suspicious_face is not None:
        best_idx = frame_indices[fake_probs.index(max_fake_prob)]
        print(f"\n   Most suspicious face:")
        print(f"   Frame #{best_idx} at {best_idx/fps:.1f}s")
        print(f"   Fake probability: {max_fake_prob*100:.1f}%")
        run_gradcam(model, suspicious_face, device, SAVE_DIR)

    timestamps = [frame_indices[i] / fps for i in range(len(frame_indices))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor("#0a0a0f")

    colors = ["#ff003c" if p > 0.5 else "#00d4ff" for p in fake_probs]
    ax1.bar(timestamps, fake_probs, color=colors, width=0.15, alpha=0.8)
    ax1.axhline(y=0.5, color="#ffaa00", linestyle="--",
                linewidth=1.5, label="Fake threshold (0.5)")
    ax1.set_facecolor("#0f172a")
    ax1.set_xlabel("Time (seconds)", color="white")
    ax1.set_ylabel("Fake Probability", color="white")
    ax1.set_title(f"Frame-by-Frame Face Analysis — {verdict}",
                  color="#ff003c" if verdict == "FAKE" else "#00d4ff",
                  fontsize=14, fontweight="bold")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#0f172a", labelcolor="white")
    ax1.set_ylim(0, 1)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#334155")

    real_count = total_frames - fake_count
    if fake_count > 0 and real_count > 0:
        wedges, texts, autotexts = ax2.pie(
            [real_count, fake_count],
            labels=["REAL", "FAKE"],
            colors=["#00d4ff", "#ff003c"],
            autopct="%1.1f%%",
            startangle=90
        )
        for t in texts + autotexts:
            t.set_color("white")
    elif fake_count == 0:
        ax2.pie([1], labels=["100% REAL"], colors=["#00d4ff"])
    else:
        ax2.pie([1], labels=["100% FAKE"], colors=["#ff003c"])

    ax2.set_facecolor("#0f172a")
    ax2.set_title("Face Frame Distribution", color="white", fontsize=12)

    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, "video_analysis.png"),
                dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()
    plt.close()

    return verdict, fake_percent


# ── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   DEEPSHIELD — Deepfake Video Detector")
    print("   (with Face Detection)")
    print("=" * 50)

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("\n   Enter path to your video file: ").strip().strip('"')

    if not os.path.exists(video_path):
        print(f"\n   ERROR: File not found: {video_path}")
        sys.exit(1)

    print(f"\n   Video: {video_path}")
    verdict, fake_percent = predict_video(video_path)