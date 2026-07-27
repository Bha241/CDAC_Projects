# 🖼️ PoseAnalyzer & Aesthetic Score Predictor

An AI-powered computer vision and deep learning system designed for analyzing human body poses, evaluating posture symmetry/alignment, and predicting image aesthetic scores on a scale of 0 to 100.

---

## 🌟 Key Features

- **Deep Learning Aesthetic Scorer**: Uses a custom **ResNet-50** architecture integrated with **RoI Align (`torchvision.ops.roi_align`)** to fuse global scene context with localized human bounding box features.
- **Pose & Posture Analysis**: Employs **MediaPipe Pose Landmarker** to analyze upper/lower body keypoints, measuring shoulder alignment, spine tilt, left/right reach symmetry, and framing.
- **Interactive Web App**: Built with **Streamlit** for real-time photo uploads and immediate aesthetic score predictions.
- **End-to-End Pipeline**: Includes complete dataset generation, pose landmark extraction, model training notebooks, and inference utilities.

---

## 📂 Project Structure

```text
PoseAnalyzer/
├── app.py                         # Streamlit web application dashboard
├── scoring_code_snippet.py        # MediaPipe pose & face landmark extraction & posture scoring logic
├── run_notebook.py                # Helper script for running training/evaluation notebooks
├── requirements.txt               # Required Python packages
├── DataCollection_Poses.ipynb     # Notebook for downloading & extracting pose landmarks into CSV format
├── AestheticScorer_Training.ipynb # PyTorch training pipeline notebook for the Aesthetic Scorer model
├── poses_scored.csv               # Dataset containing landmark metrics and aesthetic scores
├── FinalModel/
│   ├── ImageScoring.py            # Image scoring pipeline script
│   ├── AestheticScorer_Training1.ipynb # Extended training notebook
│   ├── poses_scored_new.csv       # Expanded scored dataset
│   └── unsplash_1_2k_scored.csv   # Unsplash image dataset metadata & scores
└── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Bha241/CDAC_Projects.git
cd CDAC_Projects
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  \.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Checkpoint Setup

Large binary weight files (`*.pth` and `*.task`) are excluded from Git repository tracking to keep the codebase lightweight.

Before running the inference app or training scripts, ensure model checkpoints are placed in the root directory or inside `FinalModel/`:

- **Aesthetic Model Weight Candidates**:
  - `best_model.pth` or `aesthetic_scorer_checkpoint.pth` (place in `./` or `./FinalModel/`)
- **MediaPipe Landmarker Task Model**:
  - `pose_landmarker_heavy.task` (place in `./` or `./FinalModel/`)

---

## 🚀 Usage

### 1. Launch the Streamlit Web Application
Run the Streamlit app to interactively upload images and obtain predicted aesthetic scores:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Pose & Posture Extraction
To run MediaPipe landmark detection and posture metrics calculation on custom images:
```python
from scoring_code_snippet import init_pose, score_body_symmetry, score_alignment

# Initialize pose landmarker
pose_model = init_pose('pose_landmarker_heavy.task')
```

### 3. Training & Dataset Generation
- Open `DataCollection_Poses.ipynb` in Jupyter Notebook/Lab to extract landmarks from new pose datasets.
- Open `AestheticScorer_Training.ipynb` to train or fine-tune the ResNet-50 + RoI Align model on custom image-score pairs.

---

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, torchvision (`ResNet-50`, `roi_align`)
- **Pose Detection**: MediaPipe Tasks (`PoseLandmarker`, `FaceDetector`), OpenCV
- **Frontend / Dashboard**: Streamlit
- **Data & Image Processing**: NumPy, Pandas, Pillow, Matplotlib

---

## 📝 License

This project is created for research and educational purposes.