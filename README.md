# 🧠 Mental Health NLP Analyzer

**Live Demo:** [Click Here](https://ishan7031mentalhealth00nlp.streamlit.app)

## 📌 Project Overview
An end-to-end NLP application that classifies text into 7 mental health categories (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder). Built to demonstrate the full ML lifecycle: data preprocessing, model training, evaluation, and deployment.

## 📊 Dataset
- **Source:** Kaggle (Sentiment Analysis for Mental Health)
- **Size:** 53,043 rows
- **Classes:** 7 (imbalanced)
- **Cleaning:** Removed 362 null values and 1,608 duplicates.

## 🛠️ Technology Stack
- **Data:** Pandas, NumPy, Scikit-learn
- **Models:** TF-IDF + Logistic Regression, TF-IDF + Linear SVM, DistilBERT (Hugging Face)
- **UI:** Streamlit, Plotly
- **Deployment:** Streamlit Community Cloud
- **Model Hosting:** Hugging Face Models

## 📈 Model Performance (Test Set)
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 75.2% | 69.8% |
| **DistilBERT** | **81.1%** | **75.6%** |

## ⚠️ Limitations & Error Analysis
- The model lacks an "Anger" class. Violent or aggressive text is often misclassified as "Suicidal" due to overlapping vocabulary (e.g., "kill").
- The "Personality Disorder" class has an F1 of 0.53 due to severe class imbalance in the training data.
- **Ethical Disclaimer:** This is an educational prototype, not a medical diagnostic tool.

## 🚀 How to Run Locally
1. Clone the repo: `git clone https://github.com/Ishan13-dev/mental-health-nlp.git`
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `.venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app/app.py`

## 📂 Project Structure
(Describe your folder structure here)