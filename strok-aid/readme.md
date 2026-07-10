# strok-aid

Feedforward neural network for early stroke risk prediction from patient clinical indicators, built with PyTorch.

---

## Dataset

**Stroke Prediction Dataset** — [fedesoriano, Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

| Property | Value |
|---|---|
| Rows | 5,110 |
| Features | 11 clinical indicators |
| Target | Binary (stroke / no stroke) |
| Class imbalance | ~20:1 (no stroke : stroke) |

**Features:** age, gender, hypertension, heart disease, ever married, work type, residence type, average glucose level, BMI, smoking status

---

## Pipeline

### 1. Preprocessing
- Dropped `id` column
- Median imputation for missing `bmi` values
- Removed single row with `gender = 'Other'`
- Label encoding for categorical features
- Train/test split (80/20) with stratification

### 2. Class Imbalance Handling
- Applied **SMOTE** exclusively on training data to synthesize minority class samples
- Test set kept at original distribution to reflect real-world conditions

### 3. Feature Scaling
- `StandardScaler` fitted on training data only, applied to both train and test

---

## Model Architecture — StrokAidNetV3

```
Input (11) → Linear(128) → BatchNorm → LeakyReLU(0.1) → Dropout(0.4)
           → Linear(64)  → BatchNorm → LeakyReLU(0.1) → Dropout(0.4)
           → Linear(32)  → BatchNorm → LeakyReLU(0.1) → Dropout(0.3)
           → Linear(16)  → BatchNorm → LeakyReLU(0.1) → Dropout(0.2)
           → Linear(1)   → Sigmoid
```

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Weight decay | 1e-3 |
| Scheduler | CosineAnnealingLR |
| Epochs | 150 (early stopping, patience=15) |
| Batch size | 64 |
| Decision threshold | 0.35 |

### Architecture Decisions
- **LeakyReLU** over ReLU — prevents dying neuron problem on deep layers
- **BatchNorm** at every block — stabilizes training on small tabular dataset
- **Dropout** decreasing with depth — stronger regularization early, lighter near output
- **AdamW** — cleaner weight decay separation than vanilla Adam
- **Threshold 0.35** — lowered from default 0.5 because missing an actual stroke case (false negative) is clinically worse than a false alarm; chosen via empirical threshold sweep maximizing stroke class F1

---

## Results

| Metric | Value |
|---|---|
| ROC-AUC | **0.7702** |
| Overall Accuracy | 76% |
| Stroke Recall | 58% |
| No-Stroke Precision | 97% |

### Classification Report

```
              precision    recall  f1-score   support

   No Stroke       0.97      0.77      0.86       972
      Stroke       0.11      0.58      0.19        50

    accuracy                           0.76      1022
   macro avg       0.54      0.67      0.52      1022
weighted avg       0.93      0.76      0.82      1022
```

### Confusion Matrix

|  | Predicted: No Stroke | Predicted: Stroke |
|---|---|---|
| **Actual: No Stroke** | 744 | 228 |
| **Actual: Stroke** | 21 | 29 |

### Model Versions Tried

| Version | Key Changes | AUC | Stroke Recall | Val Acc |
|---|---|---|---|---|
| V1 | Baseline — 64→32→16, ReLU, Adam | ~0.76 | ~40% | 0.73 |
| V2 | Wider — 128→64→32→16, stronger dropout | ~0.76 | ~45% | 0.75 |
| V3 | LeakyReLU, AdamW, CosineAnnealing, early stopping | **0.7702** | 58% | 0.80 |


> **Note:** ROC-AUC is the primary metric here. On heavily imbalanced medical data, accuracy is misleading — a model predicting "no stroke" always would achieve ~95% accuracy but zero clinical value.

---

## Threshold Analysis

A sweep from 0.10 to 0.89 was performed to find the optimal decision threshold by maximizing stroke class F1. The analysis confirmed **0.35** as the best balance between recall (catching actual stroke cases) and precision (avoiding excessive false alarms).

---

## Usage

### Run Notebook
Download dataset from Kaggle, place CSV in `data/`, then run `strok_aid_notebook.ipynb` top to bottom. Trained model and scaler will be saved to `models/`.


## Limitations

- Dataset is small (5,110 rows, only 249 stroke cases) — model performance is bounded by data quantity
- SMOTE on mixed categorical/numeric data is an approximation; SMOTENC would be more principled
- Stroke recall of 58% means ~42% of actual stroke cases are still missed — not production-ready without further data
- Label encoding imposes artificial ordering on categorical variables; one-hot encoding would be more theoretically sound
