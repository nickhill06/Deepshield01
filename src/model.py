import torch
import torch.nn as nn
import timm


class DeepShieldModel(nn.Module):

    def __init__(self, pretrained=True):
        super(DeepShieldModel, self).__init__()

        # load ViT-Base — num_classes=0 removes original head
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0
        )

        # ViT-Base outputs 768 features
        feature_dim = self.backbone.num_features  # 768

        # our classifier: 768 → 256 → 2
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        features = self.backbone(x)        # → (batch, 768)
        output = self.classifier(features) # → (batch, 2)
        return output


if __name__ == "__main__":
    print("⏳ Testing model...")
    model = DeepShieldModel(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print(f"✅ Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")