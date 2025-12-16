**Repository Structure**

This repository contains all the code, notebooks, and datasets used throughout the book. Each file maps directly to a section and is meant to be run alongside the corresponding chapter.

Nothing here is decorative. Every file exists for a reason.

**Notebooks**

The three notebooks below correspond to the three main sections of the book.

- SECTION_1.ipynb: Tree based models and classification.

- SECTION_2.ipynb: Regression models.

- SECTION_3.ipynb: Perceptron, logistic regression and neural networks.


**Datasets**

We work with five datasets throughout the project:

- section_1_risk_dataset.csv: Used in Section 1 to train and analyze tree based models.

- section_2_regression_dataset.csv: Used in Section 2 for regression models.

For Section 3, which focuses on churn and classification under uncertainty, we use three separate datasets that represent different internal data sources.

- section_3_classification_call_center_interactions.csv: Call center activity and behavioral signals.

- section_3_classification_form_dataset.csv: Data collected from user submitted forms.

- section_3_classification_policy_data.csv: Policy level and contractual information.

In the first section of the book we create a small flask API that can receive a request for classifiction and returns the label of one or several users:

**Python Files**

These files support the notebooks and show how models move beyond experimentation.

- etl.py: Contains the full data preprocessing logic that handles cleaning, missing values, categorical encoding, and schema alignment so the same transformations can be applied during training and at inference time.

- flask_app_classification.py: A Flask application that loads a trained classification model and serves predictions through an API.


## Usage

This repository is meant to be run locally using Python 3.11 inside a virtual environment.

### 1. Install Python 3.11

Check your current Python version:

```bash
python3 --version
```

If Python 3.11 is not installed, install it and confirm:

```bash
python3.11 --version
```

### 2. Clone the repository

```bash
git clone <REPO_URL>
cd <REPO_FOLDER>
```

### 3. Create a virtual environment

Create a virtual environment named `venv` using Python 3.11:

```bash
python3.11 -m venv venv
```

### 4. Activate the virtual environment

macOS or Linux:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Verify the correct Python is being used:

```bash
which python
python --version
```

### 5. Upgrade pip and packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 6. Install dependencies

```bash
pip install -r requirements.txt
```

### 7. Run the notebooks

```bash
jupyter notebook
```

### 8. Optional: run the Flask API

```bash
python flask_app_classification.py
```

### 9. Deactivate the environment

```bash
deactivate
```
