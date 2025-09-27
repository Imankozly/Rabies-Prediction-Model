import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# streamlit run /Users/shryqb/PycharmProjects/PythonProject/bachlor/some_running/iman_project/app.py
# --- Load model and encoders ---
model = joblib.load("/Users/shryqb/PycharmProjects/PythonProject/bachlor/some_running/iman_project/DB_model_IMAN.pkl")
cat_encoders = joblib.load("/Users/shryqb/PycharmProjects/PythonProject/bachlor/some_running/iman_project/categorical_encoders.pkl")
num_encoders = joblib.load("/Users/shryqb/PycharmProjects/PythonProject/bachlor/some_running/iman_project/numerical_scaler.pkl")
df = pd.read_excel('/Users/shryqb/PycharmProjects/PythonProject/bachlor/some_running/iman_project/Rabies__Weather__War_Combined_1.4.25.xlsx')


# --- Streamlit page config ---
st.set_page_config(
    page_title="Rabies Multi-Target Prediction",
    layout="wide",
    page_icon="🦠"
)

# --- Custom CSS ---
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(to right, #3399ff, #66ccff);
    color: #333;
}

/* Titles and headers */
h1, h2, h3 {
    color: #004d99;
    font-family: 'Arial', sans-serif;
    font-weight: bold;
}

/* Sidebar style */
.stSidebar .css-1d391kg {
    background-color: #ccffcc;
    padding: 20px;
    border-radius: 10px;
}

/* Buttons */
.stButton>button {
    background-color: #007700;
    color: white;
    font-weight: bold;
    padding: 0.6em 1.2em;
    border-radius: 8px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #004d00;
}

/* DataFrame display */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- Input UI ---
st.title("Rabies Multi-Target Prediction Dashboard")
st.header("Predict New Record")

# --- Collect inputs ---
year = st.number_input("Year", min_value=1900, max_value=2030, value=2025)
animal_species = st.selectbox("Animal Species", sorted(df['Animal Species'].dropna().unique()))
rabies_species = st.selectbox("Rabies Species", sorted(df['Rabies Species'].dropna().unique()))
settlement = st.selectbox("Settlement", sorted(df['Settlement'].dropna().unique()))
x_coord = st.number_input("x coordinate", value=0.0)
y_coord = st.number_input("y coordinate", value=0.0)
region_weather = st.selectbox("Region Weather", sorted(df['Region_Weather'].dropna().unique()))
avg_temp = st.number_input("Avg Temperature", value=25.0)
monthly_precip = st.number_input("Monthly Precipitation (mm)", value=50.0)
rainy_days = st.number_input("Rainy Days", value=5)
war_in_israel = st.selectbox("War in Israel", ["No", "Yes"])

# --- Create input DataFrame ---
feature_names = ['Year', 'Animal Species', 'Rabies Species', 'Settlement', 'x', 'y',
                 'Region_Weather', 'Avg Temperature', 'Monthly Precipitation (mm)',
                 'Rainy Days', 'War in Israel']

input_df = pd.DataFrame([[
    year, animal_species, rabies_species, settlement, x_coord, y_coord,
    region_weather, avg_temp, monthly_precip, rainy_days, war_in_israel
]], columns=feature_names)

# --- Encode & scale ---
categorical_features = ['Year','Animal Species', 'Rabies Species', 'Settlement', 'Region_Weather','War in Israel']
numerical_features = ['x', 'y', 'Avg Temperature', 'Monthly Precipitation (mm)', 'Rainy Days']

for col in categorical_features:
    if col in cat_encoders:
        input_df[col] = cat_encoders[col].transform(input_df[col].astype(str))
if num_encoders is not None:
    input_df[numerical_features] = num_encoders.transform(input_df[numerical_features])

st.subheader("Input Data")
st.dataframe(input_df)

input_df['War in Israel'] = input_df['War in Israel'].map({'Yes': 1, 'No': 0})

# --- Prediction ---
if st.button("Predict"):
    try:
        st.subheader("Predicted Values with Confidence")
        if hasattr(model, "predict_proba"):
            proba_list = model.predict_proba(input_df)
            target_names = ['Region', 'Month']

            for i, target in enumerate(target_names):
                probs = proba_list[i][0]

                # לוקח את הקטגוריות המקוריות מה-Dataset
                categories = df[target].dropna().unique()

                # יוצר DataFrame
                df_probs = pd.DataFrame({
                    "Category": categories,
                    "Probability (%)": probs * 100
                })

                # מיון מהגבוה לנמוך
                df_probs = df_probs.sort_values(by="Probability (%)", ascending=False)

                # מציג את החיזוי הראשון (הגבוה ביותר)
                top_category = df_probs.iloc[0]['Category']
                top_prob = df_probs.iloc[0]['Probability (%)']
                st.write(f"**Top Prediction for {target}: {top_category} → {top_prob:.2f}%**")

                # --- Bar chart של כל ההסתברויות ---
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    x="Category",
                    y="Probability (%)",
                    data=df_probs,
                    palette="bright",
                    ax=ax
                )
                ax.set_title(f"{target} Prediction Probabilities", fontsize=14, fontweight='bold')
                ax.set_xlabel("Category", fontsize=12)
                ax.set_ylabel("Probability (%)", fontsize=12)
                ax.set_ylim(0, 100)
                plt.xticks(rotation=45, ha='right')
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

        else:
            prediction = model.predict(input_df)
            st.success(f"Prediction: {prediction}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

