# Steel Defect Detection - Stage 1


This project builds a machine learning model to detect defects in steel surfaces using image classification. It includes EDA, model training, parameter tuning, evaluation, and deployment steps.

For full technical details, experiments, and deployment steps, see: 
[PROJECT-DETAIL.md](PROJECT-DETAIL.md)

---

## 1. Problem Description and Project Overview

In manufacturing, product quality control is a critical part of the production process. Before products are shipped to customers, inspections are required to ensure that quality and specifications meet customer requirements. In steel manufacturing, surface defects can reduce product quality and lead to customer complaints.

Due to the high speed and continuous nature of steel production, manual inspection is not practical. Therefore, cameras are installed at the production line output to capture images of steel surfaces for automated inspection.

To address this problem, this project applies machine learning to analyze captured steel surface images. This approach helps operators identify and mark defective steel more efficiently, making the inspection process faster, more consistent, and objective.

The complete system is designed as a two-stage pipeline.  
- Stage 1 performs binary classification to determine whether a steel surface is defective or non-defective.  
- Stage 2 (future work) further classifies the defect type and localizes the defect region using segmentation.

This project focuses on Stage 1, which establishes a solid baseline by detecting defective steel images using binary classification. Stage 2 will be developed in future work.



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
