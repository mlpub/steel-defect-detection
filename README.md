# Steel Defect Detection - Stage 1


This project builds a machine learning model to detect defects in steel surfaces using image classification. It includes EDA, model training, parameter tuning, evaluation, and deployment steps.

For full technical details, experiments, and deployment steps, see: 
[PROJECT-DETAIL.md](PROJECT-DETAIL.md)

---

## 1. Project Overview

Steel surface inspection is critical for quality control in manufacturing. Manual inspection is slow and error-prone. This project uses computer vision to automate defect detection from images, helping operators quickly identify defective steel.

Main steps:
* Data exploration & preprocessing
* Model training (MobileNet V2)
* Parameter tuning & experiments
* Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
* Deployment (FastAPI, Docker, Cloud Run)

---

## 2. Dataset

Source: [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection)
* 12,568 images (1600x256 px)

---

## 3. Best Model Result

Final model: MobileNet V2 (fine-tuned)
* Input size: 384x384
* Optimizer: Adam
* Augmentation: flips
* Highest F1 and ROC-AUC

Validation metrics:
| Metric    | Value |
| --------- | ----- |
| F1 Score  | 0.93  |
| Precision | 0.94  |
| Recall    | 0.92  |
| ROC AUC   | 0.98  |

---

## 4. How to Run (Local)

### Environment Setup
Follow in [PROJECT-DETAIL.md – Section 3: Environment Setup](PROJECT-DETAIL.md#3-environment-setup)


### Train the model

```
cd src
uv run train.py
```

### Run FastAPI (inference)

```
cd src
uv run predict.py
```

API endpoint: `http://localhost:9696/predict`

---

## 5. Docker & Cloud

### Build image
```
docker build -t steel-defect-app .
```

### Run container
```
docker run -p 9600:9696 steel-defect-app
```

API available at: `http://localhost:9600/predict`

---

## 6. Project Structure

```
steel-defect-detection/
│ README.md
│ PROJECT-DETAIL.md
│ pyproject.toml
│ uv.lock
│ ...
├── data/
│     train_images/
│     *.csv
├── images/
├── notebooks/
├── gcp/
├── src/
```

---

## 7. Full Report

See [PROJECT-DETAIL.md](PROJECT-DETAIL.md) for all EDA, experiments, charts, and deployment steps.
