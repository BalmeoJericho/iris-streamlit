import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ✅ Set Streamlit page config first
st.set_page_config(page_title="Iris Flower Predictor", layout="centered")

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
target_names = dict(enumerate(iris.target_names))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# UI
st.title("🌸 Iris Flower Classification")
st.write("Enter the flower measurements to predict the species.")
st.write("Jericho Neil S. Balmeo")

# 🔤 Text input fields
sepal_length = st.text_input("Sepal Length (cm)")
sepal_width = st.text_input("Sepal Width (cm)")
petal_length = st.text_input("Petal Length (cm)")
petal_width = st.text_input("Petal Width (cm)")

# Predict button
if st.button("Predict Species"):
    try:
        # Convert all inputs to floats
        values = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]
        input_df = pd.DataFrame([values], columns=iris.feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        species = target_names[prediction]
        st.success(f"🌼 Predicted Species: **{species.capitalize()}**")
    except ValueError:
        st.error("❌ Please enter valid numeric values in all fields.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Model performance section
with st.expander("Model Performance on Test Set"):
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: **{acc:.2f}**")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))
