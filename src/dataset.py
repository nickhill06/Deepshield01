import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd

# Label mapping — Real=0, Fake=1
LABEL_MAP = {"Real": 0, "Fake": 1}
ID_TO_LABEL = {0: "Real", 1: "Fake"}


def get_transforms(mode="train"):
    """
    mode = "train" → random augmentations to make model robust
    mode = "val"   → only resize and normalize, no randomness
    """
    if mode == "train":
        return transforms.Compose([
            # ViT requires exactly 224x224
            transforms.Resize((224, 224)),
            # 50% chance to flip image horizontally
            transforms.RandomHorizontalFlip(),
            # slightly change brightness/contrast
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # convert PIL image to PyTorch tensor (0-255 becomes 0.0-1.0)
            transforms.ToTensor(),
            # normalize using ImageNet stats (because ViT was pretrained on ImageNet)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset — loads images one by one during training
    Must have __len__ and __getitem__ methods
    """

    def __init__(self, dataframe, transform=None):
        # dataframe has two columns: image_path and label
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        # returns total number of images
        return len(self.df)

    def __getitem__(self, idx):
        # called by PyTorch to get ONE image by index
        row = self.df.iloc[idx]

        # open image with Pillow, convert to RGB (ensures 3 channels)
        image = Image.open(row["image_path"]).convert("RGB")

        # apply transforms (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        label = row["label"]
        return image, label


def build_dataframe(folder):
    """
    Walks through Train/Real and Train/Fake folders
    Builds a DataFrame with columns: [image_path, label]
    """
    records = []

    for class_name, label in LABEL_MAP.items():
        class_folder = os.path.join(folder, class_name)

        if not os.path.exists(class_folder):
            print(f"⚠️ Folder not found: {class_folder}")
            continue

        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_folder, filename)
                records.append({
                    "image_path": image_path,
                    "label": label
                })

    df = pd.DataFrame(records)
    print(f"📊 Found {len(df)} images")
    print(f"   Real: {len(df[df['label']==0])}")
    print(f"   Fake: {len(df[df['label']==1])}")
    return df


# Test this file directly
if __name__ == "__main__":
    df = build_dataframe("data/raw/Dataset/Train")
    print(df.head())