# 🛒 E-commerce Review Helpfulness Predictor

## 📋 Problem Statement
In e-commerce, user reviews are often ranked by recency or star rating, rather than **utility**. New buyers are often forced to skim through thousands of reviews, many of which are low-value (e.g., "Nice product" or "Good"). 

**The Goal:** Build a supervised machine learning model that predicts whether a newly written review will be marked as "Helpful" by future users, based only on its text content and metadata. This allows the platform to prioritize high-quality, informative content, improving the customer decision-making process.

---

## 🛠️ The Approach
To solve this binary classification problem (1 = Helpful, 0 = Not Helpful), I implemented a full end-to-end Machine Learning pipeline:

### 1. Data Cleaning & Labeling
* **Dataset:** Amazon Fine Food Reviews (568k reviews).
* **Filtering:** Removed reviews with 0 total votes to ensure the model learns from verified community feedback.
* **Labeling:** Defined a review as "Helpful" ($y=1$) if $> 50\%$ of users voted it as helpful; otherwise ($y=0$).

### 2. NLP & Feature Engineering
To transform human language into a mathematical coordinate space ($X$ matrix), I used:
* **Text Preprocessing:** Lowercasing, punctuation removal via Regex, and stop-word filtering using `NLTK`.
* **TF-IDF Vectorization:** Converted text into 7,000 features using **Unigrams and Bigrams**. This captures context like "not good" or "flavor aroma" rather than just isolated words.
* **Metadata Integration:** Engineered a `ReviewLength` feature using `scipy.sparse.hstack`, as longer reviews tend to correlate with higher detail and utility.

### 3. Handling Class Imbalance
The dataset was heavily skewed toward helpful reviews (majority class). To prevent the model from simply "guessing" the majority class, I applied:
* **Balanced Class Weights:** Adjusted the Logistic Regression loss function to penalize errors on the minority class ("Not Helpful") more heavily.

---

## 🔬 Model Selection & Evaluation
I compared multiple supervised learning models to find the best balance between overall accuracy and the ability to detect unhelpful reviews.

### Hyperparameter Tuning
Used `GridSearchCV` to find the optimal regularization strength.
* **Best Parameters:** `{'C': 10, 'solver': 'liblinear', 'class_weight': 'balanced'}`

### Final Metrics:
| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 73% |
| **Precision (Class 1)** | 82% |
| **Recall (Class 0)** | 44% |
| **F1-Score (Weighted)** | 73% |

---

## 📈 Key Insights
By inspecting the model's coefficients ($\beta$ weights), I discovered specific linguistic patterns:
* **Helpful Predictors:** Words describing long-term usage ("years", "always") and sensory specificity ("flavored coffee", "consistency").
* **Unhelpful Predictors:** Generic praise ("great buy", "thank amazon") or short expressions of frustration ("annoying", "doesnt").

---

## 🚀 How to Use
1. **Clone the Repo:** ```bash
   git clone [https://github.com/yourusername/helpful_review_prediction.git](https://github.com/yourusername/helpful_review_prediction.git)
