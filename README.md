## Fake News Detection on Social Media

### Project Overview
This project uses machine learning and natural language processing techniques to classify news text as **Fake** or **Real**. It uses TF-IDF to convert text into numerical features and Logistic Regression to make predictions.

### Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
- Streamlit

### Dataset
- Total records: 4,000
- Fake news records: 2,000
- Real news records: 2,000

### Model Evaluation
The model was evaluated on a held-out test split containing 1,000 records.

| Metric | Fake | Real |
|---|---:|---:|
| Precision | 0.90 | 0.88 |
| Recall | 0.88 | 0.90 |
| F1-score | 0.89 | 0.89 |

- **Test accuracy:** 88.9%
- **Test records:** 1,000

These results describe performance on this dataset's test split. They should not be interpreted as a guarantee of accuracy on real-world news.

### Web Application
An interactive Streamlit application allows users to enter news text and view the model's prediction.

**Live Demo:**  
https://fake-news-detection-6t67dpp2gwto3tuj9gp2hw.streamlit.app/

### Limitations
- The model classifies text patterns; it does not independently verify news against trusted sources.
- Predictions can be incorrect, especially for unfamiliar topics, satire, or misleading headlines.
- The dataset may contain biases or labeling errors.
- The prediction should not be treated as proof that a news story is true or false.

### Future Improvements
- Evaluate on an independent dataset of recent news.
- Improve data quality and verify labels.
- Test performance on different topics and writing styles.
- Explore additional machine learning models and features.

### GitHub Repository
https://github.com/rohitshinde9403/fake-news-detection
