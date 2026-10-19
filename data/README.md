
# Dataset

Place the processed dataset file `news.csv` in this folder.

The CSV must contain two columns:
- `text`: news article text
- `label`: `fake` or `real`

Dataset source: https://huggingface.co/datasets/Arko007/fake-news-dataset

The dataset labels may contain errors or bias. Check the dataset license and attribution requirements before redistribution.

## Experimental Results

The Logistic Regression model was evaluated on a test set of 1,000 records from a dataset containing 4,000 records.

- **Test Accuracy:** 88.9%
- **Fake class F1-score:** 0.89
- **Real class F1-score:** 0.89
- **Text Vectorization:** TF-IDF
- **Machine Learning Algorithm:** Logistic Regression

These results apply only to the current dataset sample and test split. Dataset quality and label accuracy may affect performance.

## Limitations

- The model predicts labels based on patterns learned from the training data.
- Predictions are not proof that a news story is true or false.
- News claims should be verified using reliable sources.
