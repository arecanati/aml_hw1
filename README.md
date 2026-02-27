# AML HW1: Heart Disease Classification

This repository contains an end-to-end machine learning workflow for binary heart disease classification using the UCI Cleveland dataset. The project covers raw data inspection, cleaning, feature engineering, model training, evaluation, and feature-impact analysis in a single reproducible notebook.

## Project Overview

The notebook builds a binary target from the original heart disease labels and compares several models on the same train/test split:

- Bernoulli Naive Bayes
- Gaussian Naive Bayes
- Linear Regression used as a classifier
- Ridge Regression
- Lasso Regression

The workflow also includes:

- missing-value analysis and train-only imputation
- categorical encoding
- interpretable engineered features
- confusion matrices, metric comparison plots, and ROC curves
- feature ablation to compare pre/post engineered features

## Dataset

- Raw data: `data/raw/processed.cleveland.data`
- Metadata: `data/raw/heart-disease.names`
- Processed output: `data/processed/heart_cleveland_clean.csv`

The task is binary classification:

- `0` = no heart disease
- `1` = heart disease present

## Repository Structure

```text
notebooks/aml_hw1.ipynb          Main analysis notebook
data/raw/                        Raw Cleveland dataset and metadata
data/processed/                  Saved processed dataset
reports/                         Report artifacts
requirements.txt                 Minimal pinned dependencies
```

## How to Run

Clone the repository and run the notebook locally:

```bash
git clone https://github.com/arecanati/aml_hw1.git
cd aml_hw1

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name aml_hw1 --display-name "aml_hw1"
jupyter notebook
```

Then open `notebooks/aml_hw1.ipynb`, select the `aml_hw1` kernel, and run all cells in order.

## What the Notebook Produces

Running the notebook generates:

- raw data inspection summaries
- missing-value diagnostics and imputation checks
- processed feature table
- model metrics and confusion matrices
- metric comparison plots
- ROC/AUC curves
- threshold analysis for Linear Regression
- feature ablation results for engineered features

## Reproducibility

The project includes a minimal pinned `requirements.txt` so the notebook can be recreated in a clean environment with the same core package versions.
