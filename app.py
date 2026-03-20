"""
🐝 Bee Diseases Classification — Streamlit Interface
=====================================================
Upload a bee hive image and classify it into one of six categories
using a pre-trained Swin Transformer or EfficientNetV2-S model.

Classes:
    0 - Ant Problems
    1 - Small Hive Beetles
    2 - Healthy
    3 - Robbed Hive
    4 - Missing Queen
    5 - Varroa
"""

import tempfile
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

try:
    import timm
except ImportError:
    timm = None

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = {
    0: "Ant Problems",
    1: "Small Hive Beetles",
    2: "Healthy",
    3: "Robbed Hive",
    4: "Missing Queen",
    5: "Varroa",
}

CLASS_EMOJIS = {
    0: "🐜",
    1: "🪲",
    2: "✅",
    3: "🏚️",
    4: "👑",
    5: "🦟",
}

CLASS_DESCRIPTIONS = {
    0: "Ant infestation detected in the hive. Ants can steal honey and disturb the colony.",
    1: "Small hive beetles found. These parasites damage combs, honey, and pollen.",
    2: "The hive appears healthy with no visible signs of disease or pests.",
    3: "Signs of a robbed hive. Other bees or wasps may have attacked and stolen honey.",
    4: "The colony may be queenless. Without a queen, the colony cannot sustain itself.",
    5: "Varroa mite infestation detected. Varroa destructor is one of the most serious bee threats.",
}

NUM_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────
# Model Definitions
# ──────────────────────────────────────────────


class BeeEfficientNet(nn.Module):
    """EfficientNetV2-S + Custom classifier head."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        self.backbone = models.efficientnet_v2_s(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class BeeSwinTransformer(nn.Module):
    """Swin Transformer Small with custom classification head."""

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        dropout=0.2,
        model_name="swin_small_patch4_window7_224",
    ):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for Swin Transformer. "
                "Install it with: pip install timm"
            )
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )
        embed_dim = self.backbone.num_features  # 768 for Swin-S

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

tta_transforms = [
    val_transform,
    transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    ),
    transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    ),
]


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────


@torch.no_grad()
def predict_single_image(model, image: Image.Image, device: torch.device, use_tta: bool = True):
    """Run inference on a single PIL image, optionally with TTA."""
    model.eval()
    tfms = tta_transforms if use_tta else [val_transform]

    all_probs = []
    for tfm in tfms:
        tensor = tfm(image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)[0]  # shape (num_classes,)
    return avg_probs


def load_model(model_type: str, checkpoint_path: str, device: torch.device):
    """Load a model from a checkpoint file."""
    if model_type == "Swin Transformer":
        model = BeeSwinTransformer(num_classes=NUM_CLASSES)
    else:
        model = BeeEfficientNet(num_classes=NUM_CLASSES)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────


def find_model_files():
    """Search for .pt / .pth model files in the project directory."""
    root = Path(__file__).resolve().parent
    patterns = ["*.pt", "*.pth"]
    found = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    return sorted(found)


def main():
    st.set_page_config(
        page_title="🐝 Bee Diseases Classification",
        page_icon="🐝",
        layout="wide",
    )

    # ── Header ──
    st.title("🐝 Bee Diseases Classification")
    st.markdown(
        "Upload a bee hive image to classify it into one of **6 categories** "
        "using a deep learning model trained on the BeeImage dataset."
    )

    # ── Sidebar — Model Configuration ──
    with st.sidebar:
        st.header("⚙️ Model Settings")

        model_type = st.selectbox(
            "Model Architecture",
            ["Swin Transformer", "EfficientNetV2-S"],
            help="Swin Transformer achieved 98.74% accuracy (best). "
            "EfficientNetV2-S achieved 98.55%.",
        )

        use_tta = st.checkbox(
            "Use Test-Time Augmentation (TTA)",
            value=True,
            help="Averages predictions over multiple augmentations for more robust results.",
        )

        st.divider()

        # Model checkpoint
        st.subheader("📦 Model Checkpoint")

        # Check for existing model files
        existing_models = find_model_files()

        load_method = st.radio(
            "Load model from:",
            ["Upload checkpoint file", "Select from project"],
            index=0,
        )

        checkpoint_path = None

        if load_method == "Upload checkpoint file":
            uploaded_model = st.file_uploader(
                "Upload .pt or .pth file",
                type=["pt", "pth"],
                help="Upload your trained model checkpoint.",
            )
            if uploaded_model is not None:
                tmp_path = Path(tempfile.gettempdir()) / uploaded_model.name
                tmp_path.write_bytes(uploaded_model.getvalue())
                checkpoint_path = str(tmp_path)
        else:
            if existing_models:
                selected = st.selectbox(
                    "Available checkpoints",
                    existing_models,
                    format_func=lambda p: p.name,
                )
                checkpoint_path = str(selected)
            else:
                st.warning("No .pt or .pth files found in the project directory.")

        st.divider()
        st.markdown(
            "**📊 Model Performance**\n\n"
            "| Model | Accuracy |\n"
            "|-------|----------|\n"
            "| Swin Transformer | 98.74% |\n"
            "| EfficientNetV2-S | 98.55% |\n"
        )

    # ── Main area — Image Upload ──
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("📸 Upload Image")
        uploaded_image = st.file_uploader(
            "Choose a bee hive image…",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.caption(f"Image size: {image.size[0]}×{image.size[1]} pixels")

    with col_result:
        st.subheader("🔍 Classification Result")

        if uploaded_image is None:
            st.info("👆 Upload an image to get started.")
        elif checkpoint_path is None:
            st.warning(
                "⚠️ **No model checkpoint loaded.**\n\n"
                "Please upload or select a model checkpoint file (.pt / .pth) "
                "in the sidebar.\n\n"
                "Model checkpoints can be obtained by training the models "
                "using the notebooks in the `Results/` folder, or downloaded "
                "from the associated Kaggle datasets."
            )
        else:
            # Run inference
            classify_btn = st.button(
                "🚀 Classify Image", type="primary", use_container_width=True
            )

            if classify_btn:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                st.caption(f"Using device: `{device}`")

                with st.spinner("Loading model…"):
                    try:
                        model = load_model(model_type, checkpoint_path, device)
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {e}")
                        st.stop()

                with st.spinner(
                    "Running inference" + (" with TTA…" if use_tta else "…")
                ):
                    probs = predict_single_image(model, image, device, use_tta=use_tta)

                # Display results
                pred_idx = int(np.argmax(probs))
                pred_name = CLASS_NAMES[pred_idx]
                pred_emoji = CLASS_EMOJIS[pred_idx]
                pred_conf = float(probs[pred_idx]) * 100

                # Top prediction
                if pred_conf >= 90:
                    confidence_color = "green"
                elif pred_conf >= 70:
                    confidence_color = "orange"
                else:
                    confidence_color = "red"

                st.markdown(
                    f"### {pred_emoji} {pred_name}\n"
                    f"**Confidence:** :{confidence_color}[{pred_conf:.1f}%]"
                )

                st.markdown(f"_{CLASS_DESCRIPTIONS[pred_idx]}_")

                st.divider()

                # All class probabilities
                st.markdown("**All Class Probabilities:**")
                sorted_indices = np.argsort(probs)[::-1]
                for idx in sorted_indices:
                    name = CLASS_NAMES[idx]
                    emoji = CLASS_EMOJIS[idx]
                    prob = float(probs[idx])
                    st.progress(prob, text=f"{emoji} {name}: {prob * 100:.2f}%")

    # ── Footer ──
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
        "🐝 Bee Diseases Classification — "
        "Swin Transformer & EfficientNetV2-S | "
        "BeeImage Dataset | "
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
