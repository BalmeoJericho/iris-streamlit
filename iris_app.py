import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Iris Table Classifier", layout="centered")

# Load and prepare the model
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
target_names = dict(enumerate(iris.target_names))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

st.title("🌼 Iris Flower Predictor - Table Input")
st.write("Fill in the flower measurements in the table below. Leave the **Class Label** blank — the model will predict it.")

# Template DataFrame
input_df = pd.DataFrame({
    "Sepal Length": [5.1, 6.2],
    "Sepal Width": [3.5, 2.9],
    "Petal Length": [1.4, 4.3],
    "Petal Width": [0.2, 1.3],
    "Class Label": ["", ""]
})

# Editable Table
edited_df = st.data_editor(input_df, num_rows="dynamic", use_container_width=True)

# Predict All Button
if st.button("Predict All"):
    try:
        # Drop 'Class Label' before feeding into model
        features_df = edited_df.drop(columns=["Class Label"])

        # Convert to float just in case (Streamlit may pass strings)
        features_df = features_df.astype(float)

        # Scale and predict
        scaled_input = scaler.transform(features_df)
        predictions = model.predict(scaled_input)
        predicted_species = [target_names[p] for p in predictions]

        # Show results
        edited_df["Predicted Species"] = predicted_species
        st.success("Predictions complete!")
        st.dataframe(edited_df, use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Model evaluation section
with st.expander("Model Performance"):
    y_pred = model.predict(scaler.transform(X_test))
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.2f}**")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))
