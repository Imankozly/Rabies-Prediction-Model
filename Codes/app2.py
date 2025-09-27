import streamlit as st
import pandas as pd
import joblib
import matplotlib as plt
from sklearn.tree import plot_tree
import io

# ✅ חייב להיות ראשון!
st.set_page_config(page_title="Rabies Prediction", layout="centered")

# ================== עיצוב דינמי ==================
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa, #c3cfe2); font-family: 'Arial', sans-serif; }
    h1 { color: #2c3e50; text-align: center; font-size: 3rem; font-weight: bold; }
    h2, h3 { color: #34495e; }
    div.stButton > button:first-child { background-color: #2980b9; color: white; font-size: 1.1rem; padding: 10px 24px; border-radius: 8px; border: none; transition: background-color 0.3s ease; }
    div.stButton > button:first-child:hover { background-color: #3498db; }
    div[data-baseweb="select"] > div { border-radius: 8px; border: 1px solid #2980b9; }
    </style>
    """,
    unsafe_allow_html=True
)



def compute_similarity(df: pd.DataFrame, inp: pd.DataFrame, columns: list):
    """
    מחשבת דמיון בין רשומה חדשה לבין כל הדאטה ב-DataFrame.

    פרמטרים:
    df       : DataFrame עם הנתונים הקיימים
    inp      : DataFrame עם רשומה אחת לחיזוי
    columns  : רשימת עמודות להשוואה

    מחזירה DataFrame עם עמודת 'similarity' ממוינת מהגבוה לנמוך
    """
    similarities = []

    for _, row in df.iterrows():
        score = 0
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # נורמליזציה לפי טווח העמודה
                max_val = df[col].max()
                score += 1 - abs(row[col] - inp[col].values[0]) / (max_val if max_val != 0 else 1)
            else:
                # categorical comparison
                score += (row[col] == inp[col].values[0])
        # ממוצע הדמיון על כל העמודות שנבחרו
        similarities.append(score / len(columns))

    df['similarity'] = similarities
    return df.sort_values('similarity', ascending=False)


# ================== הגדרות ==================
DATA_PATH = "/Users/shryqb/PycharmProjects/PythonProject/bachlor/some_running/iman_project/Rabies__Weather__War_Combined_1.4.25.xlsx"
MODEL_PATH = "final_model_gradient_boosting.pkl"
OHE_PATH = "preprocessing_onehot_encoder.pkl"
SCALER_PATH = "preprocessing_scaler.pkl"
TARGET_ENCODERS_PATH = "preprocessing_target_encoders.pkl"

label_cols = ['Animal Species', 'Rabies Species', 'Settlement', 'Region_Weather']
target_cols = ['Region', 'Month']
num_cols = ['x', 'y', 'Avg Temperature', 'Monthly Precipitation (mm)', 'Rainy Days']
extra_cols = ['War in Israel', 'Year']  # עמודות נוספות שהמודל דורש

# ================== טעינת המודל והנירמולים ==================
df = pd.read_excel(DATA_PATH)
model = joblib.load(MODEL_PATH)
ohe = joblib.load(OHE_PATH)
scaler = joblib.load(SCALER_PATH)
target_encoders = joblib.load(TARGET_ENCODERS_PATH)

# ================== רשימות ייחודיות עבור selectbox ==================
animal_species_list = sorted(df['Animal Species'].dropna().unique())
rabies_species_list = sorted(df['Rabies Species'].dropna().unique())
settlement_list = sorted(df['Settlement'].dropna().unique())
region_weather_list = sorted(df['Region_Weather'].dropna().unique())



# ================== כותרת ==================
#st.set_page_config(page_title="Rabies Prediction", layout="centered")
st.title("🐶 Rabies / Weather / War Prediction")
st.markdown("הזן נתונים חדשים לקבלת תחזית עבור **Region** ו־**Month**")

# ================== טופס קלט ==================
with st.form("input_form"):
    st.subheader("✍️ הזן פרטי רשומה חדשה")

    # בחירה מתוך רשימות
    animal_species = st.selectbox("Animal Species", animal_species_list)
    rabies_species = st.selectbox("Rabies Species", rabies_species_list)
    settlement = st.selectbox("Settlement", settlement_list)
    region_weather = st.selectbox("Region Weather", region_weather_list)
    war_in_israel = st.selectbox("War in Israel", ["Yes", "No"])

    # מספריים
    x = st.number_input("x", value=0.0)
    y = st.number_input("y", value=0.0)
    avg_temp = st.number_input("Avg Temperature", value=20.0)
    precipitation = st.number_input("Monthly Precipitation (mm)", value=50.0)
    rainy_days = st.number_input("Rainy Days", value=10.0)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2025)

    submitted = st.form_submit_button("🔮 Make Prediction >> ")

# המרה ל־0/1
war_in_israel_val = 1 if war_in_israel == "Yes" else 0
# בניית DataFrame יחיד
input_df = pd.DataFrame([{
    'Animal Species': animal_species,
    'Rabies Species': rabies_species,
    'Settlement': settlement,
    'Region_Weather': region_weather,
    'x': x,
    'y': y,
    'Avg Temperature': avg_temp,
    'Monthly Precipitation (mm)': precipitation,
    'Rainy Days': rainy_days,
    'War in Israel': war_in_israel_val,
    'Year': year
}])


# ================== חיזוי ==================
if submitted:
    try:
        # --- OneHot לקטגוריות ---
        encoded = ohe.transform(input_df[label_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(label_cols), index=input_df.index)

        # --- נירמול למספריים ---
        scaled_nums = scaler.transform(input_df[num_cols])
        scaled_df = pd.DataFrame(scaled_nums, columns=num_cols, index=input_df.index)

        # --- איחוד עמודות ---
        X_new = pd.concat([scaled_df, encoded_df, input_df[extra_cols]], axis=1)

        # סדר העמודות כמו במודל
        X_new = X_new[model.estimators_[0].feature_names_in_]

        # --- חיזוי ---
        y_pred = model.predict(X_new)[0]

        # --- סיכוי לכל קטגוריה ---
        region_proba = model.estimators_[0].predict_proba(X_new)[0]  # estimator[0] = Region
        month_proba = model.estimators_[1].predict_proba(X_new)[0]  # estimator[1] = Month

        # המרה חזרה לערכים מקוריים
        region_pred = target_encoders['Region'].inverse_transform([y_pred[0]])[0]
        month_pred = target_encoders['Month'].inverse_transform([y_pred[1]])[0]

        # אחוזים
        region_confidence = region_proba[y_pred[0]] * 100
        month_confidence = month_proba[y_pred[1]] * 100

        st.success(f"✅ Model Prediction: **Region = {region_pred} ({region_confidence:.2f}%), "
                   f"Month = {month_pred} ({month_confidence:.2f}%)**")

        columns_to_compare = label_cols + num_cols + extra_cols  # לדוגמה כל העמודות הרלוונטיות
        most_similar_row = compute_similarity(df, input_df, columns_to_compare)
        st.write("🟢 Most Similar Record:")
        st.dataframe(most_similar_row)



        # ================== Alerts Dictionary per Target ==================
        alerts_dict_region = {
                'Galil Golan': "⚠️ Region is 'Galil Golan', check coordinates, temperature, and precipitation values for consistency.",
                'Amakim': "⚠️ Region is 'Amakim', unusual feature values may affect prediction.",
                'Shfela Vahar': "⚠️ Region is 'Shfela Vahar', verify X/Y coordinates and weather features.",
                'Hasharon': "⚠️ Region is 'Hasharon', check numeric inputs for anomalies.",
                'Galil Maaravi': "⚠️ Region is 'Galil Maaravi', some features might be outside typical range.",
                'Negev': "⚠️ Region is 'Negev', check for extreme values in coordinates or weather data."
        }

        alerts_dict_month = {
            "January": "⚠️ Month is January, check if temperature, precipitation, and rainy days align with typical values.",
            "February": "⚠️ Month is February, unusual feature values may affect predictions.",
            "March": "⚠️ Month is March, verify coordinates and weather features for consistency.",
            "April": "⚠️ Month is April, check numeric inputs for anomalies.",
            "May": "⚠️ Month is May, some features might be outside typical range.",
            "June": "⚠️ Month is June, check for extreme values in coordinates or weather data.",
            "July": "⚠️ Month is July, unusual conditions may affect predictions.",
            "August": "⚠️ Month is August, verify temperature and precipitation values.",
            "September": "⚠️ Month is September, ensure numeric inputs are within reasonable range.",
            "October": "⚠️ Month is October, check if weather features match typical patterns.",
            "November": "⚠️ Month is November, anomalies in inputs may affect prediction.",
            "December": "⚠️ Month is December, verify coordinate and weather inputs."
        }

        # ================== Function to check alerts ==================

        st.warning(alerts_dict_month[month_pred])
        st.warning(alerts_dict_region[region_pred])

        # Feature names
        feature_names = X_new.columns.tolist()

        # GradientBoosting for Region (estimator[0])
        gb_region = model.estimators_[0].estimators_[0, 0]  # הגישה למודל פנימי של Region
        gb_month = model.estimators_[1].estimators_[0, 0]  # הגישה למודל פנימי של Region

        gb_targets = [gb_region , gb_month]
        # Feature importance for Region
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        import numpy as np
        import streamlit as st
        import matplotlib.cm as cm
        import seaborn as sns
        from scipy.stats import pearsonr, chi2_contingency
        import pandas as pd

        # ============================== plotting ==============================
        # gb_targets = רשימת המודלים של Region ו-Month, לדוגמה: model.estimators_
        target_names = ['Region', 'Month']

        for idx, i in enumerate(gb_targets):
            target = target_names[idx]
            st.subheader(f'Target Name : {target}')

            # Feature importances
            importances = i.feature_importances_
            indices = np.argsort(importances)[::-1]  # סדר יורד
            top_n = 4

            top_features = [feature_names[j] for j in indices[:top_n]]
            top_importances = importances[indices[:top_n]]

            # ===== Streamlit columns =====
            col1, col2 = st.columns(2)

            # ===== גרף Feature Importance =====
            with col1:
                plt.figure(figsize=(8, 6))
                colors = cm.viridis(np.linspace(0, 1, top_n))
                plt.barh(top_features[::-1], top_importances[::-1], color=colors)
                plt.xlabel("Feature Importance")
                plt.title(f"Top 4 Features ({target})", color='darkblue')
                st.pyplot(plt.gcf())
                plt.clf()

            # ===== Example Decision Tree =====
            with col2:
                plt.figure(figsize=(8, 6))
                plot_tree(i, feature_names=feature_names, filled=True, max_depth=3, rounded=True, fontsize=10)
                plt.title(f"Decision Tree (Depth=3) for {target}", color='darkgreen')
                st.pyplot(plt.gcf())
                plt.clf()

        # ================== Numeric Correlation ==================
        st.subheader("📊 Correlation Matrix (Numeric Features)")
        st.markdown("""
        The correlation matrix shows the pairwise **Pearson correlation coefficients** between numeric features.
        - Values close to **1** indicate a strong positive correlation.
        - Values close to **-1** indicate a strong negative correlation.
        - Values around **0** indicate little or no linear correlation.
        """)
        numeric_df = df[num_cols]
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Matrix", color='darkblue', fontsize=14)
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("📑 Pearson p-values (Numeric Features)")
        st.markdown("""
        The Pearson p-values indicate the statistical significance of the correlation between numeric features.
        - A **small p-value (typically < 0.05)** suggests that the correlation is statistically significant.
        - A **large p-value** suggests that the correlation could be due to random chance.
        - Diagonal cells are **NaN** because a feature's correlation with itself is not tested.
        """)
        pval_matrix = pd.DataFrame(np.zeros((len(num_cols), len(num_cols))), columns=num_cols, index=num_cols)
        for i, col1 in enumerate(num_cols):
            for j, col2 in enumerate(num_cols):
                pval_matrix.loc[col1, col2] = np.nan if i == j else pearsonr(numeric_df[col1], numeric_df[col2])[1]
        st.dataframe(pval_matrix.style.background_gradient(cmap="coolwarm", axis=None).format("{:.3f}"))

        # ================== Categorical Correlation (Cramér's V) ==================
        st.subheader("📊 Cramér's V (Categorical Features + Targets)")

        explain_carmer = """
        
        Cramér's V measures the strength of association between categorical variables.  
        - Values range from **0 to 1**:
          - **0** → no association  
          - **1** → perfect association  
        - Higher values indicate stronger relationships between the categories.  
        - This includes both the original categorical features and the target variables (e.g., Region, Month).
        """

        st.markdown(explain_carmer)


        categorical_cols = label_cols + target_cols
        cat_df = df[categorical_cols].dropna()



        def cramers_v(x, y):
            cmatrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(cmatrix)[0]
            n = cmatrix.sum().sum()
            phi2 = chi2 / n
            r, k = cmatrix.shape
            return np.sqrt(phi2 / min(k - 1, r - 1))


        cramers_matrix = pd.DataFrame(np.zeros((len(categorical_cols), len(categorical_cols))),
                                      index=categorical_cols, columns=categorical_cols)
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                cramers_matrix.loc[col1, col2] = 1.0 if col1 == col2 else cramers_v(cat_df[col1], cat_df[col2])

        plt.figure(figsize=(10, 8))
        sns.heatmap(cramers_matrix, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)
        plt.title("Cramér's V Correlation (Categorical Features)", color='darkgreen', fontsize=14)
        st.pyplot(plt.gcf())
        plt.clf()

        # ================== יצירת Excel ==================

        download_df = input_df.copy()
        download_df['Predicted Region'] = region_pred
        download_df['Region Confidence (%)'] = region_confidence
        download_df['Predicted Month'] = month_pred
        download_df['Month Confidence (%)'] = month_confidence

        # Feature Importances
        fi_region = pd.Series(gb_region.feature_importances_, index=feature_names, name='Region FI')
        fi_month = pd.Series(gb_month.feature_importances_, index=feature_names, name='Month FI')
        fi_df = pd.concat([fi_region, fi_month], axis=1).reset_index().rename(columns={'index': 'Feature'})


        pval_df = pval_matrix.reset_index().rename(columns={'index':'Feature1'})

        cramers_df = cramers_matrix.reset_index().rename(columns={'index':'Feature1'})
        numeric_df = df[num_cols]  # בחירת העמודות המספריות
        corr_df = numeric_df.corr()  # מטריצת קורלציה (Pearson)

        excel_buffer = io.BytesIO()
        lines = explain_carmer.split('\n')  # אם יש פסקאות
        explain_carmer_to_save = pd.DataFrame(lines, columns=['Explanation'])

        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            download_df.to_excel(writer, sheet_name='Prediction', index=False)
            most_similar_row.to_excel(writer, sheet_name='Similar row table', index=False)
            fi_df.to_excel(writer, sheet_name='Feature Importances', index=False)
            pval_df.to_excel(writer, sheet_name='Pearson p-values', index=False)
            cramers_df.to_excel(writer, sheet_name= 'Cramers V', index=False)
            pd.DataFrame(explain_carmer_to_save).to_excel(writer, sheet_name='Cramers V', index=False)
            corr_df.to_excel(writer, sheet_name='Correlation Matrix', index=True)







        st.download_button(
            label="⬇️ Download Rabies Prediction Data",
            data=excel_buffer.getvalue(),
            file_name="Rabies_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )









    except Exception as e:
        st.error(f"❌ שגיאה: {str(e)}")
