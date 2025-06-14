# 💳 Credit Card Fraud Detection using Logistic Regression

This project is a machine learning-based Streamlit web application that detects fraudulent credit card transactions using a logistic regression model. It uses the popular `creditcard.csv` dataset.

---

## 🚀 Demo

> 🎯 Try the live demo here (if hosted)  
> [https://your-streamlit-link](http://localhost:8501/)

---

## 📁 Dataset

The dataset used is from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains anonymized features of credit card transactions.

- **Rows:** 284,807 transactions
- **Features:** 30 (V1 to V28, Time, Amount)
- **Target:** `Class` → 0 = Legitimate, 1 = Fraud

---

## ⚙️ Features

- Logistic Regression model with balanced dataset (undersampling)
- Feature scaling using `StandardScaler`
- Real-time prediction interface using Streamlit
- Handles user input errors (tabs, extra spaces, wrong lengths)
- Shows prediction as Legitimate or Fraudulent

---

## 📦 Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Jupyter Notebook (for initial testing)

---

## 🧠 How It Works

1. Load and preprocess the dataset
2. Balance the classes using undersampling
3. Train a logistic regression model
4. Deploy a web interface using Streamlit
5. Accept input of 30 features and predict the class

---


## 🖥️ Running the App Locally


# Install dependencies:-
Package	Why it's needed

streamlit			#For building the interactive web app
pandas				#For data manipulation and loading the CSV
numpy				#For numerical operations and array conversions
scikit-learn			#For model training, splitting, scaling, metrics

# Run the Streamlit app
streamlit run app.py


# STRUCTURE OR FLOWCHART OF FILES:

creditcard-fraud-detection/
│
├── app.py               						# Main Streamlit application
├── Credit Card Detection Machine Learning Project with Report      	# Report of this application    				 		   
├── Credit Card Fraud Detection using Machine Learning       	    	# This is Jupiter Source file 
└── creditcard.csv             	                                      	# Dataset file


