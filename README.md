# 🧠 Breast Cancer Prediction using Neural Networks

## 📌 Project Overview

This project implements a Deep Learning model to classify breast tumors as **Malignant** or **Benign** using the Breast Cancer Wisconsin dataset.

The model is built using **TensorFlow & Keras** and achieves strong generalization performance on unseen data.

---

## 📊 Dataset Information

- Dataset: Breast Cancer Wisconsin (sklearn.datasets)
- Total Samples: 569
- Features: 30 numerical diagnostic features
- Target Classes:
  - 0 → Malignant
  - 1 → Benign

Features are computed from digitized images of breast mass cell nuclei.

---

## ⚙️ Data Preprocessing

- Converted dataset to Pandas DataFrame
- Separated features (X) and labels (Y)
- 80–20 Train-Test split
- Feature scaling using **StandardScaler**
- Validation split during training

---

## 🧠 Model Architecture

- Input Layer: 30 neurons
- Hidden Layer: 20 neurons (ReLU activation)
- Output Layer: 2 neurons (Softmax activation)

Loss Function: Sparse Categorical Crossentropy  
Optimizer: Adam  
Epochs: 20  

---

## 📈 Model Performance

- Training Accuracy: **97.03%**
- Training Loss: **0.0975**
- Test Accuracy: **96.49%**
- ROC-AUC Score: ~0.98+

The small gap between training and test accuracy indicates strong generalization and low overfitting.

---

## 📊 Evaluation Metrics

- Confusion Matrix
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC Score

These metrics provide deeper insight into classification performance beyond accuracy.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

git clone https://github.com/kartikjakhar09/Breast_Cancer_Classification.git
cd Breast_Cancer_Classification

### 2️⃣ Install Dependencies
pip install -r requirements.txt


### 3️⃣ Run Jupyter Notebook

Open the notebook file and run all cells.

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

---

## 🎯 Key Learnings

- Importance of feature scaling in Neural Networks
- Monitoring validation metrics to avoid overfitting
- Interpreting medical classification metrics (Precision, Recall, ROC-AUC)
- Building an end-to-end ML pipeline from preprocessing to inference

---

## 📬 Contact

If you're interested in collaborating on ML/AI projects, feel free to connect.
