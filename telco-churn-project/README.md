# Telco Customer Churn Prediction

This repository contains the BIL 476 Data Mining course project on predicting
customer churn using the IBM Telco Customer Churn dataset.

## Project structure

```text
telco-churn-project/
├── README.md
├── requirements.txt
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco_clean.csv
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   └── 02_preprocessing_modeling.ipynb
├── reports/
│   ├── report.tex
│   └── report.pdf
└── results/
    └── model_results.csv
```

The modeling notebook also generates the figures used by the report in the
`reports/` directory.

## Setup

Create a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Jupyter from the project root:

```bash
jupyter notebook
```

Run the notebooks in this order:

1. `notebooks/01_eda_and_cleaning.ipynb`
2. `notebooks/02_preprocessing_modeling.ipynb`

The second notebook must be rerun before updating the numerical results,
figures, discussion, and conclusion in the report.

## Method summary

- Stratified 80/20 train-test split
- Decision Tree, Gaussian Naive Bayes, k-NN, and Random Forest
- Stratified 5-fold grid search using churn-class F1-score
- One-hot encoding and k-NN scaling inside cross-validation pipelines
- SMOTENC inside cross-validation to handle mixed categorical/numeric data
- Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC evaluation

