import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_align
from PIL import Image
import os

# --- Configuration (must match notebook) ---
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_CANDIDATES = [
    os.path.join(ROOT_DIR, 'FinalModel', 'best_model.pth'),
    os.path.join(ROOT_DIR, 'best_model.pth'),
    os.path.join(ROOT_DIR, 'FinalModel', 'aesthetic_scorer_checkpoint.pth'),
    os.path.join(ROOT_DIR, 'aesthetic_scorer_checkpoint.pth'),
]

# --- Model Definition ---
class AestheticScorer(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone_layers=6,
                 roi_output_size=7, dropout=0.3):
        super().__init__()
        weights  = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        frozen = [self.layer0, self.layer1, self.layer2]
        if freeze_backbone_layers >= 4:
            frozen.append(self.layer3)
        for m in frozen:
            for p in m.parameters():
                p.requires_grad = False

        self.roi_output_size = roi_output_size
        self.global_pool     = nn.AdaptiveAvgPool2d(1)
        self.local_pool      = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(4096, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 128),  nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def extract_features(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(x)))))

    def forward(self, images, boxes_norm):
        feat = self.extract_features(images)
        fH, fW = feat.shape[2:]
        global_feat = self.global_pool(feat).squeeze(-1).squeeze(-1)

        boxes_abs = boxes_norm.clone().float()
        boxes_abs[:,1] *= fW; boxes_abs[:,3] *= fW
        boxes_abs[:,2] *= fH; boxes_abs[:,4] *= fH

        roi_feats  = roi_align(feat, boxes_abs,
                               output_size=(self.roi_output_size, self.roi_output_size),
                               spatial_scale=1.0, aligned=True)
        local_feat = self.local_pool(roi_feats).squeeze(-1).squeeze(-1)

        img_idx = boxes_abs[:,0].long()
        fused   = torch.cat([global_feat[img_idx], local_feat], dim=1)
        return self.head(fused).squeeze(1) * 100.0


# --- Image Transform ---
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


@st.cache_resource
def load_model():
    model = AestheticScorer(pretrained=True, freeze_backbone_layers=6).to(DEVICE)
    ckpt_path = next((p for p in CHECKPOINT_CANDIDATES if os.path.exists(p)), None)
    if ckpt_path is None:
        st.error(
            "Model checkpoint not found. Please ensure 'best_model.pth' is present in the project root or in the 'FinalModel' folder."
        )
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


model = load_model()


def infer_image(img_pil, bbox_norm=None):
    if model is None: return None
    model.eval()
    tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)
    if bbox_norm is None:
        bbox_norm = [0.0, 0.0, 1.0, 1.0]
    box = torch.tensor([[0.0]+bbox_norm], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        score = model(tensor, box)
    return round(score.item(), 2)


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Aesthetic Score Predictor")
st.title("🖼️ Aesthetic Score Predictor")
st.markdown("Upload an image and get its aesthetic score (0-100).")

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_pil = Image.open(uploaded_file).convert('RGB')

    if st.button("Predict Score"):
        with st.spinner("Predicting..."):
            predicted_score = infer_image(img_pil, bbox_norm=[0.0, 0.0, 1.0, 1.0])

            with col2:
                if predicted_score is not None:
                    st.metric(label="Predicted Aesthetic Score", value=f"{predicted_score:.2f}")
                else:
                    st.error("Could not predict score.")
