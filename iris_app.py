import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Iris Table Input", layout="centered")

# Load and train model
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
target_names = dict(enumerate(iris.target_names))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

st.title("🌸 Iris Flower Classification (Text-Box Table Style)")
st.write("Fill in the input fields below, arranged like a table row.")

# 👇 Create a horizontal layout to simulate a table row
st.markdown("### Table Input")
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col1:
    sepal_length = st.text_input("Sepal Length")
with col2:
    sepal_width = st.text_input("Sepal Width")
with col3:
    petal_length = st.text_input("Petal Length")
with col4:
    petal_width = st.text_input("Petal Width")
with col5:
    class_label = st.text_input("Class Label (Optional)")

# Predict button
if st.button("Predict Species"):
    try:
        values = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]
        input_df = pd.DataFrame([values], columns=iris.feature_names)
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        predicted_species = target_names[prediction]

        st.success(f"🌼 Predicted Species: **{predicted_species.capitalize()}**")
    except ValueError:
        st.error("❌ Please enter valid numeric values in all fields.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Optional: Model performance
with st.expander("Model Performance"):
    y_pred = model.predict(scaler.transform(X_test))
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.2f}**")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))
