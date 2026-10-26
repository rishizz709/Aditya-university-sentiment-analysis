# Real-Time Sentiment Analysis System

**Machine Learning Project | Aditya University**

## Abstract

The Real-Time Sentiment Analysis System analyzes student feedback and classifies comments into Positive, Negative, and Neutral categories using Natural Language Processing (NLP) and Machine Learning.

The application provides a web interface where users can enter feedback and view the predicted sentiment. The project uses text preprocessing, TF-IDF feature extraction, and a trained machine learning model.

## Objectives

- Analyze student feedback automatically.
- Clean and preprocess textual data.
- Convert text into numerical features using TF-IDF.
- Classify feedback into Positive, Negative, and Neutral sentiments.
- Compare machine learning algorithms.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Explore sentiment visualization and real-time data collection.

## Features

- Student feedback input through a web interface.
- Sentiment prediction.
- Text preprocessing.
- TF-IDF feature extraction.
- Saved machine learning model and vectorizer.
- Flask-based web application.

## Technology Stack

- Python
- Flask
- Natural Language Processing (NLP)
- TF-IDF
- Scikit-learn
- Pandas
- Joblib
- HTML, CSS, and JavaScript
- Jupyter Notebook

## Dataset

The project uses an Aditya University student feedback dataset.

The presentation identifies these attributes:

- Comment
- Sentiment
- Platform
- Aspect
- Rating
- Student type
- Year of study
- Engagement score

Only share the dataset publicly if you have permission and it contains no confidential student information.

## Methodology

1. Data collection
2. Data cleaning
3. Text preprocessing and normalization
4. TF-IDF feature extraction
5. Machine learning model training
6. Sentiment classification
7. Model performance evaluation
8. Result visualization

## Algorithms

The project presentation discusses Logistic Regression and Support Vector Machine (SVM) for sentiment classification and comparison.

The final algorithm and its performance should be confirmed from the actual training code and evaluation results.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve, where applicable

Actual performance scores will be added after model evaluation.

## Project Structure

```text
Sentiment_Analysis_Project/
├── app.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
├── Aditya_University_Sentiment_Final_Dataset.xlsx
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── sentiment_model_metadata.json
├── Sentiment_Analysis__final.ipynb
├── static/
├── templates/
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the placeholders with your actual repository details.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment on Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Run the application

```powershell
python app.py
```

Open the local URL displayed in the terminal, typically http://127.0.0.1:5000.

### 6. Retrain the model (optional)

```powershell
python train_model.py
```

## Future Enhancements

- Integrate web scraping and APIs for additional data sources.
- Add interactive sentiment charts and dashboards.
- Implement alerts for significant sentiment changes.
- Explore topic modeling.
- Support large-scale streaming text data.
- Explore advanced deep-learning models such as LSTM.

These are proposed enhancements unless they are already implemented.

## Team Members

- B. Rishi Vardhan — Data Collection and Cleaning
- Mohammed Jaffer — Data Preprocessing and EDA
- S. Hemanth Kumar — Model Building
- K. Ashok Kumar — Evaluation and Deployment

## Academic Details

- Course: Machine Learning (241AI024)
- Department: Artificial Intelligence and Machine Learning
- Institution: Aditya University

## Conclusion

This project demonstrates the use of NLP and machine learning to analyze student feedback. The application provides a foundation for understanding student opinions through automated sentiment classification.

## Disclaimer

This project is intended for academic and educational purposes. Predictions may not always capture the intended meaning of a comment.
