# **Google Colab Lab Assignment - NLP**
### Course Name: [Enter Course Name]  
### Lab Title: NLP Techniques for Text Classification  
### Student Name: [Enter Your Name]  
### Student ID: [Enter Your ID]  
### Date of Submission: [Enter Date]  
### Group Members: [Enter Names]  

---

## **Objective**
The objective of this assignment is to implement NLP preprocessing techniques and build a text classification model using machine learning techniques.

---

## **Learning Outcomes:**
- Understand and apply NLP preprocessing techniques such as tokenization, stopword removal, stemming, and lemmatization.
- Implement text vectorization techniques such as TF-IDF and CountVectorizer.
- Develop a text classification model using a machine learning algorithm.
- Evaluate the performance of the model using suitable metrics.

---

## **Assignment Instructions:**

### **Part 1: NLP Preprocessing**

1. **Dataset Selection:**
   - Choose any text dataset from [Best Datasets for Text Classification](https://en.innovatiana.com/post/best-datasets-for-text-classification) such as the SMS Spam Collection, IMDb Reviews, or any other relevant dataset.
   - Download the dataset and upload it to Google Colab.

2. **Load the Dataset:**
   - Load the dataset into a Pandas DataFrame and explore its structure (e.g., check missing values, data types, and label distribution).

3. **Text Preprocessing:**
   - Convert text to lowercase.
   - Perform tokenization using NLTK or spaCy.
   - Remove stopwords using NLTK or spaCy.
   - Apply stemming using PorterStemmer or SnowballStemmer.
   - Apply lemmatization using WordNetLemmatizer.

4. **Vectorization Techniques:**
   - Convert text data into numerical format using TF-IDF and CountVectorizer.

---

## **Code for Part 1:**

```python
from google.colab import files
uploaded = files.upload()

import pandas as pd
# Load dataset
df = pd.read_csv("dataset.csv")

# Checking for missing values
print("Missing values:\n", df.isnull().sum())

# Checking data types and distributions
print("\nData Types:\n", df.dtypes)
