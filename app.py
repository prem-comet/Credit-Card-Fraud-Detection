<<<<<<< HEAD
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import streamlit as st

# # load data
# data = pd.read_csv('creditcard.csv')

# # separate legitimate and fraudulent transactions
# legit = data[data.Class == 0]
# fraud = data[data.Class == 1]

# # undersample legitimate transactions to balance the classes
# legit_sample = legit.sample(n=len(fraud), random_state=2)
# data = pd.concat([legit_sample, fraud], axis=0)

# # split data into training and testing sets
# X = data.drop(columns="Class", axis=1)
# y = data["Class"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# # train logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # evaluate model performance
# train_acc = accuracy_score(model.predict(X_train), y_train)
# test_acc = accuracy_score(model.predict(X_test), y_test)

# # create Streamlit app
# st.title("Credit Card Fraud Detection Model")
# st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# # create input fields for user to enter feature values
# input_df = st.text_input('Input All features')
# input_df_lst = input_df.split(',')
# # create a button to submit input and get prediction
# submit = st.button("Submit")

# if submit:
#     # get input feature values
#     features = np.array(input_df_lst, dtype=np.float64)
#     # make prediction
#     prediction = model.predict(features.reshape(1,-1))
#     # display result
#     if prediction[0] == 0:
#         st.write("Legitimate transaction")
#     else:
#         st.write("Fraudulent transaction")






# //**********************Correct Code *******************//

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
=======
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
>>>>>>> 62bd1cf6164f4e02b60734f7845d0badc8a0381a
