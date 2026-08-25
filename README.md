# Parkinson's Disease Detection using Vocal Acoustic Features

## Project Information

**Problem Statement:** Parkinson’s Disease (PD) diagnosis traditionally relies on subjective clinical evaluations. Vocal impairment is an early sign of PD, presenting an opportunity for objective, machine learning-based screening.
**Objective:** To design a robust machine learning pipeline that classifies patients as "Healthy" (0) or "Parkinson's" (1) using biomedical voice measurements, utilizing Subject-Aware Cross-Validation to prevent data leakage.
**Dataset Source & Target Variable:** UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/174/parkinsons). The target variable is `status`.
**Technologies Used:** Python, Pandas, Scikit-Learn, Seaborn, Matplotlib.
**Project Workflow:** Data Loading -> Subject-Aware Holdout Split -> RFECV Feature Selection -> StandardScaler Normalization -> Model Training -> Evaluation.

## Results & Usage

**Models Implemented:** Logistic Regression, Support Vector Machine (RBF), Random Forest, K-Nearest Neighbors.

**Model-Performance Table:**
| Model | Test Accuracy | PD Precision | PD Recall |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **90.7%** | **0.89** | **1.00** |
| SVM (RBF) | 88.4% | 0.86 | 1.00 |
| Logistic Regression | 86.0% | 0.86 | 0.97 |
| KNN (k=3) | 86.0% | 0.84 | 1.00 |

**Key Visualizations & Findings:** 
RFECV successfully reduced 22 acoustic features down to the 5 most critical markers: `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `NHR`, `spread1`, and `PPE`. 

**Best Model & Conclusion:**
The Random Forest Classifier performed best, achieving 90.7% accuracy and perfectly recalling all Parkinson's patients (Recall = 1.00) in the isolated test set without data leakage.

**How to Install and Run:**
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis: `jupyter notebook notebooks/test.ipynb`

