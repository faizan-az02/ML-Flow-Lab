# ML-Flow-Lab

An MLflow-powered experimentation lab for tracking, comparing, and
managing multiple classification models on a shared dataset.

------------------------------------------------------------------------

# Overview

This repository demonstrates how to use **MLflow** to manage machine
learning experiments.\
Multiple classification models are trained on the same dataset while
MLflow tracks:

-   model parameters
-   evaluation metrics
-   experiment runs
-   model artifacts
-   model flavors

All experiments are logged under a **single MLflow experiment** called:

`mlflow-demo`

The MLflow UI allows easy comparison of models and their performance.

------------------------------------------------------------------------

# Project Structure

    ML-Flow-Lab
    │
    ├── logistic_regression_exp.py
    ├── random_forest_exp.py
    ├── gradient_boosting_exp.py
    │
    ├── run_all_experiments.py
    │
    └── README.md

------------------------------------------------------------------------

# Models Used

Three different classification models are trained on the same dataset:

  Script                       Model
  ---------------------------- ---------------------
  logistic_regression_exp.py   Logistic Regression
  random_forest_exp.py         Random Forest
  gradient_boosting_exp.py     Gradient Boosting

All scripts use the **Iris dataset from scikit-learn**.

------------------------------------------------------------------------

# MLflow Tracking

Each experiment logs the following information:

### Parameters

Example:

    model = LogisticRegression
    max_iter = 200

### Metrics

-   accuracy
-   precision
-   recall

### Artifacts

MLflow stores model artifacts such as:

    MLmodel
    model.pkl
    conda.yaml
    requirements.txt
    python_env.yaml

------------------------------------------------------------------------

# Model Flavors

The logged models automatically register MLflow flavors.

Example flavors recorded in the **MLmodel** file:

    python_function
    sklearn

These flavors allow the model to be loaded in multiple ways.

------------------------------------------------------------------------

# Running the Project

## 1 Install dependencies

    pip install mlflow scikit-learn

## 2 Start the MLflow UI

    mlflow ui

Open the interface in your browser:

    http://localhost:5000

## 3 Run experiments

Run experiments individually:

    python logistic_regression_exp.py
    python random_forest_exp.py
    python gradient_boosting_exp.py

Or run all experiments automatically:

    python run_all_experiments.py

------------------------------------------------------------------------

# MLflow Interface

Inside the MLflow UI you can:

-   compare experiment runs
-   inspect model parameters
-   analyze performance metrics
-   download model artifacts
-   review model flavors

All runs appear under the experiment:

    mlflow-demo

------------------------------------------------------------------------

# Dataset

The project uses the **Iris dataset**, a built-in dataset from
`scikit-learn`.

Features include measurements of iris flowers used to classify species.

------------------------------------------------------------------------

# Purpose

This repository provides a hands-on workflow for:

-   experiment tracking
-   model comparison
-   artifact management
-   MLflow experiment organization
-   reproducible ML pipelines

------------------------------------------------------------------------

# License

This project is intended for educational and experimentation purposes.
