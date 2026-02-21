# 🛒 E-commerce Review Helpfulness Predictor

## 📌 Project Overview
New buyers often struggle to find useful information among thousands of reviews. This project uses **Supervised Learning** to predict whether a newly written review will be marked as "Helpful" by the community based on its text content and metadata.

## ⚙️ The Workflow
1. **Data Cleaning:** Handled a dataset of 500,000+ Amazon Fine Food reviews, filtering for reviews with active community feedback.
2. **Text Preprocessing:** Used `NLTK` and `Regex` for tokenization, stop-word removal, and noise reduction.
3. **Feature Engineering:** - **TF-IDF Vectorization:** Converted text into a 7,000-dimensional matrix using Unigrams and Bigrams.
   - **Metadata Extraction:** Engineered a "Review Length" feature to capture the correlation between detail and helpfulness.
4. **Handling Imbalance:** Applied `class_weight='balanced'` to address the shortage of "Not Helpful" training examples.
5. **Hyperparameter Tuning:** Used `GridSearchCV` to find the optimal regularization strength ($C=100$).

## 💾 Data
The dataset used in this project is the **Amazon Fine Food Reviews**. 
Due to its size, the CSV file is not included in this repository. 
You can download it from Kaggle or via `kagglehub`:
`path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")`

## 📊 Key Results
- **Model:** Logistic Regression (Optimized)
- **Accuracy:** 73%
- **Key Insight:** The model learned that specific sensory details (e.g., "flavor aroma") and long-term usage (e.g., "years") are stronger predictors of helpfulness than generic positive sentiment (e.g., "great buy").

## 🛠️ Tools Used
- **Python** (Pandas, NumPy, Scikit-Learn)
- **NLP:** NLTK, TF-IDF
- **Visualization:** Matplotlib, WordCloud