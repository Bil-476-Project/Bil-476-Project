# Telco Customer Churn Prediction

This repository contains a data mining project developed for the **BIL 476 Data Mining** course. The project predicts whether a telecommunications customer will churn and compares four classification algorithms:

- Decision Tree
- Gaussian Naive Bayes
- k-Nearest Neighbors (k-NN)
- Random Forest

Each algorithm is evaluated both with and without **SMOTENC** to examine how class balancing affects churn prediction.

## Dataset

The project uses the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) published on Kaggle.

The dataset contains **7,043 customer records** and **21 columns**, including:

- Customer demographics
- Account and contract information
- Subscribed services
- Monthly and total charges
- Churn status

The `customerID` column is removed before modeling because it is an identifier rather than a predictive feature. Eleven blank values in `TotalCharges` are converted to missing values and handled during preprocessing.

## Repository Structure

```text
Bil-476-Project/
├── README.md
└── telco-churn-project/
    ├── requirements.txt
    ├── data/
    │   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
    │   └── telco_clean.csv
    ├── notebooks/
    │   ├── 01_eda_and_cleaning.ipynb
    │   └── 02_preprocessing_modeling.ipynb
    ├── reports/
    │   ├── BIL476_Report_Zeynep_Ay_231404035.tex
    │   ├── BIL476_Report_Zeynep_Ay_231404035.pdf
    │   ├── figures_confusion_matrices.png
    │   ├── figures_f1_comparison.png
    │   ├── figures_feature_importance.png
    │   ├── figures_pr_curves.png
    │   └── figures_roc_curves.png
    └── results/
        └── model_results.csv
```

## Installation

The project was developed with **Python 3.10**.

1. Clone the repository:

   ```bash
   git clone https://github.com/Bil-476-Project/Bil-476-Project.git
   cd Bil-476-Project/telco-churn-project
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```bash
   .venv\Scripts\activate
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

## Running the Project

Run the notebooks from the project root in the following order:

1. `notebooks/01_eda_and_cleaning.ipynb`
2. `notebooks/02_preprocessing_modeling.ipynb`

The first notebook performs exploratory data analysis and cleaning. The second notebook runs preprocessing, hyperparameter selection, model training, evaluation, and figure generation.

Rerun the second notebook before updating the numerical results, figures, discussion, or conclusion in the report.

## Methodology

- Stratified 80/20 train-test split
- Stratified 5-fold cross-validation
- Grid search using the churn-class F1-score
- One-hot encoding for categorical features
- Scaling for k-NN
- Model-specific preprocessing inside cross-validation pipelines
- SMOTENC applied inside cross-validation to prevent data leakage
- Evaluation with Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes + SMOTENC | 0.727 | 0.492 | 0.778 | **0.602** | 0.804 | 0.575 |
| Random Forest + SMOTENC | 0.764 | 0.544 | 0.671 | 0.601 | 0.830 | 0.616 |
| Decision Tree | 0.798 | 0.635 | 0.567 | 0.599 | 0.828 | 0.621 |
| k-NN + SMOTENC | 0.727 | 0.491 | 0.762 | 0.597 | 0.808 | 0.560 |
| Naive Bayes | 0.703 | 0.465 | **0.810** | 0.591 | 0.819 | 0.603 |
| Decision Tree + SMOTENC | 0.728 | 0.492 | 0.738 | 0.590 | 0.791 | 0.499 |
| Random Forest | **0.803** | **0.662** | 0.524 | 0.585 | **0.839** | **0.653** |
| k-NN | 0.780 | 0.589 | 0.564 | 0.577 | 0.827 | 0.601 |

The results show that no model is best for every objective:

- **Naive Bayes + SMOTENC** achieved the highest F1-score.
- **Naive Bayes without SMOTENC** achieved the highest recall and detected the largest proportion of churn customers.
- **Random Forest without SMOTENC** achieved the highest accuracy, precision, ROC-AUC, and PR-AUC.

Therefore, model selection depends on the business objective. A company that prioritizes detecting more possible churn customers may prefer a higher-recall model, while a company that wants fewer false alarms may prefer Random Forest.

## Reproducibility

The random seeds, train-test split, preprocessing steps, cross-validation strategy, and selected hyperparameters are defined in the notebooks. The complete evaluation output is also available in:

```text
results/model_results.csv
```

## AI Tool Use

ChatGPT and Claude were used for coding, debugging, and improving parts of the experimental pipeline. ChatGPT also assisted with report development, language revision, structure, and LaTeX formatting. The project workflow, outputs, tables, figures, citations, and final content were reviewed and verified by the student.

## Author

**Zeynep Ay**  
TOBB University of Economics and Technology  
Artificial Intelligence Engineering  
Email: [zay@etu.edu.tr](mailto:zay@etu.edu.tr)

