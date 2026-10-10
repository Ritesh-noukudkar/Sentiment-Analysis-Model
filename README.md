# 🎬 Sentiment Analysis using NLP & Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF-red)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

## 📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) technique used to identify the emotional tone or sentiment expressed in textual data.

In this project, a Machine Learning model is developed to classify IMDb movie reviews into two sentiment categories:

- 😊 Positive
- 😞 Negative

The project follows an end-to-end Machine Learning workflow, starting from raw review data and continuing through text preprocessing, feature extraction using TF-IDF, model training, evaluation, model serialization, and prediction of new user reviews.

The trained model and TF-IDF vectorizer are also saved using Pickle so that they can be reused without retraining the model.

---

## 🎯 Project Objectives

- Analyze movie review text data.
- Perform exploratory data analysis (EDA).
- Clean and preprocess textual data.
- Remove unnecessary characters and stopwords.
- Convert text into numerical features using TF-IDF.
- Train a Machine Learning classification model.
- Evaluate the performance of the trained model.
- Save the trained model using Pickle.
- Save the TF-IDF vectorizer for future predictions.
- Create a custom sentiment prediction function.
- Test the model using new user-provided reviews.

---

## 📊 Dataset

This project uses the **IMDb Movie Review Dataset**.

The dataset contains:

- **50,000 movie reviews**
- **25,000 training reviews**
- **25,000 testing reviews**
- Two sentiment categories:
  - Positive
  - Negative

### Dataset Features

| Column | Description |
|---|---|
| `review` | Movie review text |
| `sentiment` | Sentiment label: Positive or Negative |

---

# 🔄 Project Workflow

```text
IMDb Dataset
     ↓
Data Loading
     ↓
Exploratory Data Analysis
     ↓
Text Preprocessing
     ↓
Stopword Removal
     ↓
TF-IDF Vectorization
     ↓
Train-Test Split
     ↓
Logistic Regression
     ↓
Model Prediction
     ↓
Model Evaluation
     ↓
Save Model & Vectorizer
     ↓
Custom Review Prediction
```

---

# 🧹 1. Text Preprocessing

Raw movie reviews contain unnecessary information such as HTML tags, punctuation, URLs, numbers, and common stopwords.

The following preprocessing techniques were applied:

### Lowercase Conversion

```python
text = text.lower()
```

### HTML Tag Removal

```python
text = re.sub(r'<.*?>', '', text)
```

### URL Removal

```python
text = re.sub(r'http\S+|www\S+', '', text)
```

### Punctuation Removal

```python
text = text.translate(
    str.maketrans('', '', string.punctuation)
)
```

### Number Removal

```python
text = re.sub(r'\d+', '', text)
```

### Stopword Removal

```python
stop_words = set(stopwords.words('english'))
```

---

# 🔢 2. TF-IDF Feature Extraction

Machine Learning algorithms cannot directly understand raw text.

Therefore, the cleaned reviews are converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The implementation uses:

```python
TfidfVectorizer(max_features=5000)
```

Example:

```python
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['clean_review'])
```

---

# ✂️ 3. Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

- **80% → Training Data**
- **20% → Testing Data**

---

# 🤖 4. Machine Learning Model

## Logistic Regression

Logistic Regression is used as the classification algorithm.

In this project:

```text
0 → Negative
1 → Positive
```

### Model Implementation

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

---

# 📈 5. Model Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

### Classification Report

```python
print(classification_report(y_test, y_pred))
```

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

---

# 📊 Model Performance

Use the actual result obtained from the Jupyter Notebook.

| Metric | Result |
|---|---|
| Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF |
| Maximum Features | 5,000 |
| Train-Test Split | 80:20 |
| Accuracy | **Add your actual accuracy here** |

> Replace the accuracy placeholder with the actual value from your notebook.

---

# 💾 6. Model Serialization

The trained Machine Learning model is saved using Python's Pickle library.

```text
model/
└── sentiment_model.pkl
```

### Save Model

```python
import pickle

with open("../model/sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)
```

---

# 🔢 7. Save TF-IDF Vectorizer

The fitted TF-IDF vectorizer is also saved because new reviews must be transformed using the same vectorizer used during training.

```python
with open("../model/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)
```

Saved files:

```text
model/
├── sentiment_model.pkl
└── tfidf_vectorizer.pkl
```

---

# 🔮 8. Custom Review Prediction

The prediction pipeline is:

```text
User Review
     ↓
Text Cleaning
     ↓
TF-IDF Transformation
     ↓
Trained Logistic Regression Model
     ↓
Positive / Negative
```

### Prediction Function

```python
def predict_sentiment(review):

    cleaned = clean_text(review)

    vectorized = loaded_vectorizer.transform([cleaned])

    prediction = loaded_model.predict(vectorized)[0]

    if prediction == 1:
        return "Positive 😊"
    else:
        return "Negative 😞"
```

---

# 🧪 9. Testing with Custom Reviews

### Positive Review

```python
review = "This movie was absolutely fantastic. I loved the story and acting."

print(predict_sentiment(review))
```

Expected type of output:

```text
Positive 😊
```

### Negative Review

```python
review = "This movie was terrible and boring. I completely hated it."

print(predict_sentiment(review))
```

Expected type of output:

```text
Negative 😞
```

---

# 📁 10. Project Structure

```text
Sentiment_Analysis/
│
├── data/
│   └── IMDB Dataset.csv
│
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook/
│   └── sentiment_analysis.ipynb
│
├── report/
│   └── Sentiment_Analysis_Project_Report.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🛠️ 11. Technologies Used

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Natural Language Processing
- NLTK
- Regular Expressions

### Machine Learning
- Scikit-learn
- Logistic Regression
- TF-IDF

### Data Visualization
- Matplotlib
- Seaborn

### Development Environment
- Jupyter Notebook
- VS Code

### Model Serialization
- Pickle

---

# ⚙️ 12. Installation

Clone the repository:

```bash
git clone https://github.com/RiteshNoukudkar/Sentiment_Analysis.git
```

Navigate to the project:

```bash
cd Sentiment_Analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ 13. Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebook/sentiment_analysis.ipynb
```

Run the notebook cells sequentially.

The notebook contains:

- Data loading
- EDA
- Data preprocessing
- TF-IDF vectorization
- Model training
- Model evaluation
- Model saving
- Custom prediction

---

# 📌 14. Key Learning Outcomes

This project provided practical experience in:

- Natural Language Processing
- Text preprocessing
- Exploratory Data Analysis
- Feature engineering
- TF-IDF vectorization
- Binary classification
- Logistic Regression
- Model evaluation
- Confusion Matrix
- Precision, Recall and F1-score
- Model serialization
- Custom prediction functions
- Python project organization

---

# 🚀 15. Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Word embeddings
- LSTM
- GRU
- BERT
- Transformer-based models
- Streamlit web application
- Flask or FastAPI API
- Cloud deployment

---

# ⚠️ 16. Limitations

- The model is trained specifically on movie reviews.
- Performance may differ on product reviews, social media posts, or customer feedback.
- Logistic Regression may not capture complex contextual relationships between words.
- Sarcasm and ambiguous language can be difficult to classify.
- Prediction quality depends on the training dataset.

---

# 🎓 17. Project Outcome

The project demonstrates an end-to-end Sentiment Analysis workflow using NLP and Machine Learning.

```text
Raw Review
     ↓
Preprocessing
     ↓
TF-IDF
     ↓
Machine Learning Model
     ↓
Positive / Negative
```

The trained Logistic Regression model and TF-IDF vectorizer are stored separately using Pickle, allowing the solution to be reused for future predictions.

---

# 👨‍💻 Author

## Ritesh Maruti Noukudkar

**B.Tech. Computer Science & Engineering**

### Skills Demonstrated

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Machine Learning
- Natural Language Processing
- Data Analysis
- Data Visualization

---

# 📌 Internship Project

This project was developed as part of a **Data Science Internship** to gain practical experience in Natural Language Processing, Machine Learning, text classification, model evaluation, and building a reusable machine learning solution.

---

# 📄 Project Documentation

The repository contains documentation covering:

- Project methodology
- Dataset
- Text preprocessing
- Feature extraction
- Model development
- Model evaluation
- Model serialization
- Custom prediction
- Results
- Future improvements

---

# ⭐ Conclusion

This Sentiment Analysis project demonstrates how Natural Language Processing and Machine Learning can be combined to automatically classify textual reviews.

By using text preprocessing, TF-IDF feature extraction, and Logistic Regression, the project converts unstructured movie reviews into machine-readable features and predicts their sentiment.

The project also demonstrates practical model reusability by saving both the trained model and TF-IDF vectorizer, allowing predictions on new reviews without retraining.

---

## ⭐ If you found this project useful, consider giving the repository a star!
