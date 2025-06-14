import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ------------------- Load and Prepare Data -------------------
data = pd.read_csv('creditcard.csv')

# Separate classes
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample to balance
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Train/test split
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2)

# ------------------- Train Model -------------------
# Use pipeline with scaling and logistic regression
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Accuracy (Optional to show later)
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# ------------------- Streamlit App -------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Enter feature values to predict if a transaction is **Legitimate** or **Fraudulent**.")

st.caption("🔢 Input all feature values **separated by commas**, e.g. `0.1, -1.2, 2.3, ...`")

input_data = st.text_area("Enter 30 feature values (comma-separated):")

if st.button("Submit"):
    clean_input = input_data.replace('\t', ',').strip()  # replace tabs with commas
    try:
        values = [float(val.strip()) for val in clean_input.split(',')]

        if len(values) != X.shape[1]:
            st.error(f"❌ You must enter exactly {X.shape[1]} values (you entered {len(values)}).")
        else:
            input_array = np.array(values).reshape(1, -1)
            prediction = model.predict(input_array)

            if prediction[0] == 0:
                st.success("✅ This is a Legitimate transaction.")
            else:
                st.error("🚨 This is a Fraudulent transaction!")

    except ValueError:
        st.error("❌ Please enter only valid numeric values separated by commas.")

