# Fake News Detection on Social Media

A beginner-friendly machine-learning project that classifies a news text as **fake** or **real** using Natural Language Processing (NLP), TF-IDF text features, and Logistic Regression.

> **Important limitation:** This project is an educational baseline, not a fact-checking authority. It can learn patterns in its training data and may be wrong. A model label or confidence score does not prove a claim is true or false.

## Project objectives

- Prepare labeled news text for machine learning.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Try a prediction on new text.

These objectives align with the college report's stated goal of text preprocessing, machine-learning-based classification, and evaluation. This repository is a new implementation based on those objectives; it does not claim to be the original code used in the college project.

## Tech stack

- Python
- Pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- joblib

## Repository structure

```text
fake-news-detection/
├── data/
│   └── news.csv              # Add a properly labeled dataset here
├── fake_news_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset format

Create `data/news.csv` with two columns: `text` and `label`. Each label must be `fake` or `real`.

```csv
text,label
"Example news text from a verified dataset",real
"Example text labeled fake by the dataset source",fake
```

The lines above are format examples only, not verified news. Do not use them as evidence or for meaningful model evaluation. Use a reputable, appropriately licensed dataset and preserve its attribution/license information. Make sure both classes have enough examples; a larger balanced dataset is recommended.

## Setup

Python 3.10 or newer is recommended.

### 1. Create and activate a virtual environment (optional but recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add your dataset

Put your CSV at `data/news.csv`, or pass a different path using `--data`.

### 4. Train and evaluate

```bash
python fake_news_detection.py --train
```

The script prints accuracy, precision, recall, F1-score, and a confusion matrix, then saves the trained model as `fake_news_model.joblib`.

### 5. Predict a text

After training:

```bash
python fake_news_detection.py --predict "Type the news text you want to classify here"
```

The result is a **model prediction only**. Verify claims with reliable primary sources and professional fact-checkers.

## How it works

1. Load and validate the labeled CSV.
2. Split the data into training and test sets.
3. TF-IDF converts text into numerical features.
4. Logistic Regression learns patterns from the training examples.
5. Evaluate on held-out test examples.
6. Save the trained pipeline and use it for a sample prediction.

## Responsible use and limitations

- Training-data quality and labeling directly affect results.
- A random train/test split may not measure performance on new events or platforms.
- Language, satire, context, and changing misinformation patterns can cause errors.
- Do not present the output as a definitive truth verdict.
- Do not upload private, copyrighted, or restricted datasets without permission.
- Do not commit secrets, API keys, or private user data.

## Future improvements

- Compare Logistic Regression with Linear SVM.
- Add cross-validation and error analysis.
- Test performance on different topics and languages.
- Build a small web interface after validating the model.
- Explore source verification and explainability without treating model output as proof.

## Author

Rohit Shinde

## Project status

Learning project / baseline implementation. Evaluation results should be reported only after training on a suitable labeled dataset.
