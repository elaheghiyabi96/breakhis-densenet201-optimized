# breakhis-densenet201-optimized
Optimized DenseNet201 model for breast cancer binary classification on BreakHis dataset - 96.21% test accuracy with 3-phase training and TTA
🧬 BreakHis Breast Cancer Classification - DenseNet201 Optimized
<div align="center">
https://img.shields.io/badge/Python-3.9+-blue.svg
https://img.shields.io/badge/TensorFlow-2.10+-orange.svg
https://img.shields.io/badge/Keras-2.10+-red.svg
https://img.shields.io/badge/Accuracy-96.21%25-brightgreen.svg
https://img.shields.io/badge/Model-DenseNet201-9cf.svg
https://img.shields.io/badge/Dataset-BreakHis-9cf.svg
https://img.shields.io/badge/Google%20Colab-Supported-yellow.svg
https://img.shields.io/badge/License-MIT-blue.svg

Optimized DenseNet201 Model for Breast Cancer Binary Classification (Benign vs Malignant)

</div>
📌 Overview
This project implements a state-of-the-art optimized DenseNet201 model for binary classification of breast cancer histopathology images using the BreakHis dataset. Through a 3-phase training strategy with optimized fine-tuning and Test Time Augmentation (TTA), the model achieves 96.21% test accuracy, making it the best performing model among all approaches tested.

🎯 Key Results
Metric	Value
Test Accuracy (TTA)	96.21% 🏆
Test Accuracy (Original)	96.12%
Best Validation Accuracy	96.28%
Benign Precision	0.92
Benign Recall	0.96
Malignant Precision	0.98
Malignant Recall	0.96
F1-Score (Weighted)	0.96
📊 Performance Comparison
Rank	Model	Test Accuracy
🥇	DenseNet201 Optimized + TTA	96.21%
🥈	DenseNet201 Optimized (Original)	96.12%
🥉	DenseNet201 (Previous)	95.45%
4	DenseNet121	91.15%
5	Ensemble (3 Models)	90.99%
6	CNN Baseline	88.24%
📊 Dataset Overview
The BreakHis (Breast Cancer Histopathological Images) dataset consists of microscopic images of breast tumor tissues.

Feature	Description
Total Images	7,909
Classes	Benign (2,480) / Malignant (5,429)
Magnifications	40X, 100X, 200X, 400X
Image Size	224 × 224 × 3 (RGB)
Format	PNG / JPG / JPEG
Class Distribution
text
Benign    ████████████████░░░░░░░░░░░░░░░░  2,480 (31.4%)
Malignant ████████████████████████████████  5,429 (68.6%)
Note: The dataset is imbalanced (Malignant = 69%, Benign = 31%). Class weights were applied to address this.

🧠 Model Architecture
DenseNet201 Architecture
python
# DenseNet201 with Transfer Learning
base_model = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
Layer Details
text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ densenet201 (Functional)        │ (None, 7, 7, 1920)     │    18,321,984 │
│ global_average_pooling2d        │ (None, 1920)           │             0 │
│ batch_normalization             │ (None, 1920)           │         7,680 │
│ dense (Dense)                   │ (None, 512)            │       983,552 │
│ dropout (Dropout)               │ (None, 512)            │             0 │
│ dense_1 (Dense)                 │ (None, 256)            │       131,328 │
│ dropout_1 (Dropout)             │ (None, 256)            │             0 │
│ dense_2 (Dense)                 │ (None, 1)              │           257 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total Parameters: 19,444,801 (74.18 MB)
Trainable Parameters: 1,118,977 (4.27 MB)

🔧 3-Phase Training Strategy
Phase 1: Frozen Training (30 Epochs)
python
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Result: Validation Accuracy 95.78%, Loss 0.1457
Phase 2: Optimized Fine-Tuning (25 Epochs)
python
# Unfreeze last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Result: Validation Accuracy 96.28%, Loss 0.1061
Phase 3: Final Fine-Tuning (15 Epochs)
python
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Result: Test Accuracy 96.12%
📈 Advanced Data Augmentation
python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=15,
    fill_mode='reflect'
)
Training Configuration
Parameter	Value
Batch Size	32
Train/Val/Test Split	70% / 15% / 15%
Optimizer	Adam
Class Weights	Balanced (Benign: 1.59, Malignant: 0.73)
Early Stopping Patience	8
ReduceLR Patience	4
Augmentation	Advanced (rotation, brightness, flip)
🎯 Test Time Augmentation (TTA)
TTA applies multiple augmentations to each test image and averages predictions:

python
def predict_with_tta(img, n_aug=5):
    preds = [model.predict(img)[0][0]]
    for _ in range(n_aug):
        aug = apply_random_augmentation(img)
        preds.append(model.predict(aug)[0][0])
    return np.mean(preds)
TTA Result: Improved accuracy from 96.12% to 96.21%

📈 Results
Final Classification Report (TTA)
text
              precision    recall  f1-score   support
      Benign       0.92      0.96      0.94       372
   Malignant       0.98      0.96      0.97       815

    accuracy                           0.96      1187
   macro avg       0.95      0.96      0.96      1187
weighted avg       0.96      0.96      0.96      1187
Confusion Matrix
text
[[358  14]   ← Benign: 358 correct, 14 misclassified
 [ 31 784]]  ← Malignant: 784 correct, 31 misclassified
Training Progress
Phase	Best Val Accuracy	Best Val Loss
Phase 1 (Frozen)	95.78%	0.1457
Phase 2 (Fine-Tuning)	96.28%	0.1061
Phase 3 (Final FT)	96.11%	0.1099
Final Test	96.21% (TTA)	-
🚀 How to Run
Step 1: Setup Google Colab
python
from google.colab import drive
drive.mount('/content/drive')
Step 2: Dataset Preparation
Place your dataset zip file in Google Drive:

text
/content/drive/MyDrive/dataset_cancer_v1.zip
Expected folder structure:

text
dataset_cancer_v1/
└── classificacao_binaria/
    ├── 40X/
    │   ├── benign/
    │   └── malignant/
    ├── 100X/
    │   ├── benign/
    │   └── malignant/
    ├── 200X/
    │   ├── benign/
    │   └── malignant/
    └── 400X/
        ├── benign/
        └── malignant/
Step 3: Run the Code
Execute all cells in the Colab notebook. The code will:

Extract the dataset automatically

Apply advanced data augmentation

Train DenseNet201 in 3 phases

Run TTA evaluation

Generate comprehensive metrics

Save the best model

Step 4: Outputs
After training, you'll receive:

Classification Report

Confusion Matrix

Original vs TTA Accuracy

Training History

Saved Model (.keras)

Test Accuracy: 96.21% 🏆

🛠️ Requirements
bash
pip install tensorflow numpy pandas scikit-learn matplotlib
Dependencies
text
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
📁 Repository Structure
text
breakhis-densenet201-optimized/
│
├── README.md                                    # You are here
├── requirements.txt                             # Dependencies
├── densenet201_optimized.py                     # Main training script
├── densenet201_optimized.ipynb                  # Colab notebook
├── densenet201_optimized_final.keras            # Best model (96.21%)
└── .gitignore
💡 How to Use the Trained Model
python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('densenet201_optimized_final.keras')

# Load and preprocess image
img = image.load_img('sample.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]
result = "Malignant" if prediction > 0.5 else "Benign"
print(f"Prediction: {result} (confidence: {prediction:.2f})")
🔍 Interpretation of Results
What It Means	Details
High Precision (Malignant 98%)	Very low false positive rate
High Recall (Benign 96%)	Excellent at identifying Benign cases
High Recall (Malignant 96%)	Rarely misses true Malignant cases
Consistent F1-Score (0.96)	Balanced performance across both classes
Only 45 Errors	Out of 1187 test images
🎓 Key Lessons Learned
What Worked
✅ 3-Phase Training Strategy enabled gradual learning

✅ Optimized LR (5e-5) was perfect for fine-tuning

✅ Unfreezing 30 layers provided optimal flexibility

✅ Advanced Augmentation improved generalization

✅ BatchNormalization stabilized training

✅ Early Stopping with Patience=8 prevented overfitting

✅ Test Time Augmentation added slight improvement

Why This Model is the Best
Factor	Impact
DenseNet201 Architecture	More powerful than DenseNet121
3-Phase Training	Gradual, stable learning
Optimized Learning Rates	1e-3 → 5e-5 → 1e-5
Advanced Augmentation	Better generalization
30 Layers Unfrozen	More adaptability
TTA	Slight accuracy boost
📊 Model Comparison Summary
Model	Test Accuracy	Parameters	Size	Time
CNN Baseline	88.24%	9.8M	37 MB	20 min
DenseNet121	91.15%	8M	30 MB	30 min
Ensemble	90.99%	28M	110 MB	120 min
DenseNet201 (Previous)	95.45%	19.4M	74 MB	90 min
DenseNet201 Optimized	96.21%	19.4M	74 MB	90 min
⚠️ Limitations
No Patient-Level Split – Currently uses image-level split

No Cross-Validation – Only single train/val/test split

Limited Data – More images could improve robustness

Training Time – ~90 minutes in Colab

🚀 Future Improvements
Priority	Improvement	Expected Impact
1	Patient-wise Cross-Validation	More reliable evaluation
2	Fine-Tuning All Layers	+0.5-1% accuracy
3	Multi-Class Classification	8 tumor subtypes
4	Grad-CAM Visualization	Model interpretability
5	Mobile Deployment	Real-world application
📚 Dataset Citation
bibtex
@article{breakhis2016,
  title={BreakHis: A Dataset for Breast Cancer Histopathological Image Classification},
  author={Spanhol, F.A. and Oliveira, L.S. and Petitjean, C. and Heutte, L.},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2016}
}
🌐 Dataset Source
https://data.mendeley.com/datasets/jxwvdwhpc2/1

🔑 Keywords
text
Breast Cancer, Histopathology, Binary Classification, DenseNet201, Transfer Learning,
Deep Learning, BreakHis Dataset, Medical Imaging, Benign vs Malignant,
Optimized Fine-Tuning, Test Time Augmentation, TensorFlow, Keras,
Image Classification, Computational Pathology, AI in Healthcare
📝 License
This project is licensed under the MIT License.

<div align="center">
⭐ If this project helped you, don't forget to star it! ⭐

</div>
