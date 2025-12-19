## Repository Structure

This repository contains all the code, notebooks, and datasets used throughout the book. Each file maps directly to a section and is meant to be run alongside the corresponding chapter.

## Notebooks

The three notebooks below correspond to the three main sections of the book.

1. `SECTION_1.ipynb`: Tree based models and classification.
2. `SECTION_2.ipynb`: Regression models.
3. `SECTION_3.ipynb`: Perceptron, logistic regression and neural networks.

## Datasets

We work with five datasets throughout the project.

Section 1:
1. `section_1_risk_dataset.csv`: Used in Section 1 to train and analyze tree based models.

Section 2:
2. `section_2_regression_dataset.csv`: Used in Section 2 for regression models.

Section 3 focuses on churn and classification under uncertainty, so we use three separate datasets that represent different internal data sources.

3. `section_3_classification_call_center_interactions.csv`: Call center activity and behavioral signals.
4. `section_3_classification_form_dataset.csv`: Data collected from user submitted forms.
5. `section_3_classification_policy_data.csv`: Policy level and contractual information.

In Section 1 of the book we create a small Flask API that can receive a request for classification and returns the label of one or several users.

## Python Files

These files support the notebooks and show how models move beyond experimentation.

1. `etl.py`: Contains the full data preprocessing logic that handles cleaning, missing values, categorical encoding, and schema alignment so the same transformations can be applied during training and at inference time.
2. `flask_app_classification.py`: A Flask application that loads a trained classification model and serves predictions through an API.

## Usage

To run the notebooks you will have to install several packages, so the best way to do this is via a virtual env. Think of it as an independent Python env where you can install everything you need to run all the code in the project.

One caveat: all these steps worked on my machine. I am confident they will work on yours also, but this is software so I cannot guarantee it 100 percent.

Ok, let us get this party started.

### 1. Install Python 3.11

Check your current Python version:

```bash
python3 --version
```

If Python 3.11 is not installed, install it and confirm:

```bash
python3.11 --version
```

### 2. Clone or download the repository

You need a copy of this folder on your computer and there are two ways you can do it.

Option 1: clone it using the command line or any IDE option. I assume that if you choose this option you know you need the repo URL.

Option 2: simply download the repo to your computer.

### 3. Create a virtual environment

Now we will create the virtual environment. I think it will be easier if you use an IDE that supports Python. I use VS Code but if you have a different one you like, no problem.

Open the project folder. Next open a terminal. In VS Code, in the top menu bar, click Terminal and select New Terminal. In the terminal, run:

```bash
python3.11 -m venv before_machine_learning
```
I suggest python 3.11 because it was the version that worked the best with tensorflow.
### 4. Activate the virtual environment

After creating the venv in step 3, we now must activate it.

macOS or Linux:

```bash
source before_machine_learning/bin/activate
```

Windows PowerShell:

```powershell
before_machine_learning\Scripts\Activate.ps1
```

### 5. Upgrade pip and packaging tools

Before we install everything else, we update pip, the tool that allows us to download and install packages.

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 6. Install dependencies

Now we are ready. The file `requirements.txt` contains all the packages required for our project.

```bash
pip install -r requirements.txt
```
TensorFlow went through a round of updates around the time this book was published, and those updates introduced some dependency issues with NumPy. In short, we need to keep NumPy below version 2. The requirements.txt file is already configured to enforce this.

That said, I did run into a few issues on my machine, and there is a chance you might see the same thing. If that happens, do not panic. The Section 3 notebook includes a workaround that handles the problem directly. The relevant cells are clearly documented and explain exactly what is going on, so you should be able to fix it quickly if needed.

### 7. Run the notebooks

If you wish to run the notebooks, run:

```bash
jupyter notebook
```

## Flask API

Ok, now we need to speak about how we will run this API thing.

First, check the notebook for the first section. There we have a cell that creates a script called `etl.py`. It is true that the script comes with the repo, however, there you will find more detailed instructions on why we need it.

To run the Flask app, run:

```bash
python flask_app_classification.py
```

Important: if you want to run the Jupyter notebook and the Flask app at the same time, you need two terminals. These are two web applications that run on different ports, therefore they can be ran in parallel. You do not need one to run the other, meaning you do not have to have Jupyter running for the API to work.

When you run the Flask app, it will be listening on port 5000. If you want to test it, we need to send a curl request. For that, open yet another terminal. I know, I know, get used to the black and white, it will be part of your life now. Once you have the new terminal open, run the code below to perform the curl request.

### Test API

Open a new terminal and run:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "dance_disaster_rating": 7.5,
      "ticklish_level": 4,
      "eyesight": 8,
      "age": 350,
      "number_of_policies": null,
      "creature_type": "Dragon",
      "flight_status": "Yes",
      "fire_power_level": 120,
      "soda_likeness_rate": 52,
      "snack_obsession": 12,
      "rage_outburst_score": 3,
      "hearing_score": 6.2,
      "burp_burn_index": 3.4,
      "owl_ratio": 0.17,
      "grandma_boogie_index": 9.6
    }
  }'
```

That is an example of a set of features for a random customer. The response should be the classification of the three selected models plus some more information.

For more details please check the comments on the code and notebooks.
