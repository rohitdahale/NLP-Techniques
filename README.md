# NLP Text Classification Project

## Project Overview

This repository contains the implementation of a Natural Language Processing (NLP) project where we perform text classification using a variety of techniques. The goal is to preprocess text data and build a model to classify the text into different categories. We used the **Sentiment Analysis for Financial News** dataset for this assignment, which is based on financial news sentiment classification.

---

## Key Steps and Approach

### Dataset Loading and Preprocessing:
- The dataset was loaded into a Pandas DataFrame, and we began by checking for missing values, inspecting data types, and exploring the distribution of labels in the dataset.
- The dataset was processed through multiple NLP steps, including:

  - **Tokenization:** The text was split into individual words using NLTK’s `word_tokenize`.
  - **Stopwords Removal:** Common words such as "the", "is", "in" were removed using NLTK’s list of stopwords.
  - **Stemming and Lemmatization:** We applied **PorterStemmer** for stemming and **WordNetLemmatizer** for lemmatization to reduce words to their base forms (e.g., “running” to “run”).
  - **Text Vectorization:** We used **TF-IDF (Term Frequency-Inverse Document Frequency)** and **CountVectorizer** to convert the preprocessed text into numerical features suitable for machine learning.

### Model Building:
- After preprocessing, the data was split into training and testing sets (80% training, 20% testing).
- A **Logistic Regression** model was trained on the training data and tested on the test data.
- Other machine learning models like **Naïve Bayes** can be tried as alternative classifiers, but we focused on **Logistic Regression** in this case.

### Model Evaluation:
- We evaluated the model’s performance using various metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- A **Confusion Matrix** was used to visualize the model's predictions, showing the true positive, false positive, true negative, and false negative classifications.

### Final Results:
- The model performed well on the test data and was evaluated against multiple performance metrics.
- Visualizations were provided, including a confusion matrix and performance metrics, to understand how well the model generalized to unseen data.

---

## Technologies and Libraries Used

- **Python:** For the overall implementation.
  
### Libraries:
- **Pandas:** For data manipulation.
- **NLTK:** For text preprocessing tasks like tokenization, stopword removal, stemming, and lemmatization.
- **scikit-learn:** For machine learning algorithms and model evaluation.
- **Matplotlib/Seaborn:** For visualizations such as confusion matrix and performance metrics.
- **Google Colab:** Used for executing the code and notebook sharing.

---

## Challenges
- Ensuring the correct implementation of text preprocessing steps like stopword removal, stemming, and lemmatization was important for optimizing the text input to the model.
- Handling **class imbalance** and ensuring that the model performs well across all classes.

---

## Conclusion
This project helped in implementing several NLP techniques, applying them to real-world text data, and building a classification model. The performance metrics and confusion matrix results show that the model has a good understanding of the text, classifying the financial news into the correct sentiment categories.

---

## Dataset Link
- [Sentiment Analysis for Financial News - Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
