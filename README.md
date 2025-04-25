# Fingerprint Deepfake Detection using Traditional ML & Deep Learning

## Problem Statement

Fingerprint-based biometric systems are increasingly threatened by spoofing attacks such as silicone replicas, 3D-printed fingers, and AI-generated fingerprints. Traditional machine learning models often fail to generalize due to limited feature representation. This project focuses on detecting fingerprint deepfakes using both classical machine learning and deep learning approaches.

## Dataset Collection

The fingerprint datasets used are too large to host directly on GitHub. Please download them from the Google Drive link below:

**Download Datasets:** [Datasets (Google Drive)](https://drive.google.com/drive/folders/19QanCCAnysk-3JebsYqcXtaaZCz-RJcj?usp=drive_link)

### Dataset Contents

- `original_dataset.zip` – Raw fingerprint images
- `resized(224x224).zip` – Resized images for deep learning models
- `normalised.zip` – Normalized images for ML models
- `Generated_Dataset.zip` – GAN-generated fingerprint spoof images
- `Generated_Resized_Normalised/` – Final processed dataset used for training

## Preprocessing

Fingerprint images were processed using the following steps:

1. **Resizing**: All images resized to 224x224 using `resize.py`
2. **Normalization**: Pixel values normalized using `normalise.py`
3. **Feature Extraction**: Gabor, LBP, or texture features extracted using `extract_fingerprint_features.py` for ML models

## Traditional Machine Learning Models

We used traditional ML classifiers as a baseline:

- `svm.py`: Implements Support Vector Machine classifier
- `xgboost_fingerprint.py`: Implements XGBoost classifier

Limitations:
- Overfitting observed due to low feature diversity
- Poor generalization to unseen spoofing techniques

## Deep Learning Models

Given the complexity of biometric spoof detection, deep learning is more effective. The following models were implemented:

| Model             | Strengths                             | Type               |
|------------------|----------------------------------------|--------------------|
| SimpleCNN         | Lightweight, fast                      | Shallow CNN        |
| ResNet-18         | Robust deep feature extraction         | Residual CNN       |
| EfficientNet-B0   | High accuracy with efficient scaling   | Optimized CNN      |
| ViT-Small         | Captures global context and structure  | Vision Transformer |

Each model was trained on the resized and normalized datasets.


## Challenges Faced

- Overfitting in ML models due to limited feature representation
- Dataset-specific cues affecting model generalization
- Need for models that can detect evolving spoofing methods
- DL models still risk learning superficial patterns without diverse data

## Key Takeaways

1. Fingerprint spoof detection is a complex, high-stakes problem in biometric security.
2. Classical ML models struggle to generalize across spoof types.
3. Deep learning models perform better due to their ability to learn discriminative patterns.
4. Combining CNNs and ViTs provides both local and global feature understanding.
5. Dataset diversity and augmentation are critical for robust spoof detection.



