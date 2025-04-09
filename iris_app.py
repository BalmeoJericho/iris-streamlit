import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Map target to species names
target_names = dict(enumerate(iris.target_names))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Streamlit UI  
st.set_page_config(page_title="Iris Flower Predictor", layout="centered")
st.title("🌸 Iris Flower Classification")
st.write("Enter the flower measurements to predict the species.")
st.write("Jericho Neil S.Balmeo")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict Species"):
    try:
        # Prepare and scale the input
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=iris.feature_names)
        input_scaled = scaler.transform(input_data)

        # Predict and map result
        prediction = model.predict(input_scaled)[0]
        species = target_names[prediction]

        st.success(f" Predicted Species: **{species.capitalize()}**")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display model performance (optional)
with st.expander("Model Performance on Test Set"):
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: **{acc:.2f}**")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Map target to species names
target_names = dict(enumerate(iris.target_names))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Streamlit UI
st.set_page_config(page_title="Iris Flower Predictor", layout="centered")
st.title("🌸 Iris Flower Classification")
st.write("Enter the flower measurements to predict the species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# Predict button
if st.button("Predict Species"):
    try:
        # Prepare and scale the input
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=iris.feature_names)
        input_scaled = scaler.transform(input_data)

        # Predict and map result
        prediction = model.predict(input_scaled)[0]
        species = target_names[prediction]

        st.success(f" Predicted Species: **{species.capitalize()}**")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display model performance (optional)
with st.expander("Model Performance on Test Set"):
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: **{acc:.2f}**")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))
 