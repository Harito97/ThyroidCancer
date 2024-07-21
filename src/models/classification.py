import torch
import torch.nn as nn
import torchvision.models as models


class H97_EfficientNet(nn.Module):
    def __init__(
        self, num_classes: int = 3, retrainEfficientNet: bool = False, dropout=0.5
    ):
        super(H97_EfficientNet, self).__init__()
        # Load a pretrained EfficientNet model
        efficientnet = models.efficientnet_b0(pretrained=True)
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])
        self.num_layers = len(list(self.feature_extractor.children()))

        if not retrainEfficientNet:
            # Freeze the parameters in the feature extractor to not update during training
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # The input size for the first fully connected layer based on the output of EfficientNet
        self.fc1 = nn.Linear(1280, 9)
        self.fc2 = nn.Linear(9, 7)
        self.fc3 = nn.Linear(7, 3)
        self.fc4 = nn.Linear(3, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        # Flatten the tensor from [batch_size, 1280, 1, 1] to [batch_size, 1280] to match the fully connected layer
        x = torch.flatten(x, 1)

        # Pass through the dense network
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = nn.Softmax(dim=1)(x)
        return x

    def set_parameter_requires_grad(self, start_layer, end_layer):
        if start_layer > end_layer:
            print("Invalid layer index: Start layer should be less than end layer")
            return
        layers = list(self.feature_extractor.children())
        for i in range(start_layer, end_layer):
            for param in layers[i].parameters():
                param.requires_grad = True

    def get_optimizers(self, learning_rate):
        parameters_to_optimize = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(parameters_to_optimize, lr=learning_rate)


import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTFeatureExtractor


class ViTTinyModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTTinyModel, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=num_classes
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        self.num_layers = 12
        self.initial_lr = 1e-4
        self.lr_high = 10 * self.initial_lr
        self.lr_low = self.initial_lr

    def set_parameter_requires_grad(self, start_layer, end_layer):
        """Set requires_grad for layers based on their index."""
        layer_names = list(self.model.vit.named_parameters())
        for i, (name, param) in enumerate(layer_names):
            if i < start_layer:
                param.requires_grad = False
            elif start_layer <= i < end_layer:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_optimizers(self, layer_idx):
        """
        Create different optimizers for different layers.
        """
        optimizers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Split the parameter name based on '.' and check its prefix to determine the layer
                if "encoder" in name:
                    # Extract the layer index from the name
                    try:
                        # Example: name might be 'vit.encoder.layer.0.attention.self.query.weight'
                        layer_number = int(
                            name.split("encoder.layer.")[1].split(".")[0]
                        )
                    except (IndexError, ValueError):
                        layer_number = None

                    if layer_number is not None:
                        if layer_number >= layer_idx:
                            optimizers.append({"params": param, "lr": self.lr_high})
                        else:
                            optimizers.append({"params": param, "lr": self.lr_low})
                else:
                    # Handle other parameters if needed
                    optimizers.append({"params": param, "lr": self.lr_low})
        return optimizers

    def forward(self, images):
        return self.model(images).logits


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        dim=256,
        depth=12,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
    ):
        super(ViT, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
            nn.Linear(dim, dim),
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, dim)
        )

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        # Classification head
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Extract patches and apply embedding
        patches = self.patch_embedding(x)
        patches = patches.permute(0, 2, 1).contiguous()

        # Add positional encoding
        batch_size = patches.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x += self.positional_encoding

        # Apply transformer
        x = self.transformer(x)

        # Classification head
        cls_output = x[:, 0]
        output = self.fc(cls_output)

        return output
