# ================== Imports ==================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# ================== Evaluation functions ==================
def evaluate_multioutput_models(X, y, models, n_splits=5, n_runs=5):
    """Evaluate multi-output models with repeated K-Fold CV."""
    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        region_accuracies, month_accuracies = [], []

        for run in range(n_runs):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + run)

            run_region_acc, run_month_acc = [], []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                run_region_acc.append(accuracy_score(y_test['Region'], y_pred[:, 0]))
                run_month_acc.append(accuracy_score(y_test['Month'], y_pred[:, 1]))

            region_accuracies.append(np.mean(run_region_acc))
            month_accuracies.append(np.mean(run_month_acc))

        results.append({
            'Model': name,
            'Average Region Accuracy': np.mean(region_accuracies),
            'Average Month Accuracy': np.mean(month_accuracies)
        })

    return pd.DataFrame(results).sort_values(by='Average Region Accuracy', ascending=False)


def evaluate_catboost_multi_target(X, y, n_splits=5, n_runs=5):
    """Evaluate CatBoost separately for Region & Month."""
    print("\nEvaluating CatBoost (separate models for Region & Month)...")
    region_acc, month_acc = [], []

    for run in range(n_runs):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + run)

        run_region_acc, run_month_acc = [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train_region, y_test_region = y['Region'].iloc[train_index], y['Region'].iloc[test_index]
            y_train_month, y_test_month = y['Month'].iloc[train_index], y['Month'].iloc[test_index]

            model_region = CatBoostClassifier(verbose=0, random_state=42)
            model_month = CatBoostClassifier(verbose=0, random_state=42)

            model_region.fit(X_train, y_train_region)
            model_month.fit(X_train, y_train_month)

            run_region_acc.append(accuracy_score(y_test_region, model_region.predict(X_test)))
            run_month_acc.append(accuracy_score(y_test_month, model_month.predict(X_test)))

        region_acc.append(np.mean(run_region_acc))
        month_acc.append(np.mean(run_month_acc))

    avg_region = np.mean(region_acc)
    avg_month = np.mean(month_acc)
    print(f"\nCatBoost Accuracy - Region: {avg_region:.4f}, Month: {avg_month:.4f}")

    return pd.DataFrame([{
        'Model': 'CatBoost (Separate)',
        'Average Region Accuracy': avg_region,
        'Average Month Accuracy': avg_month
    }])


# ================== Load dataset ==================
df = pd.read_excel(
    "/Users/shryqb/PycharmProjects/PythonProject/bachlor/some_running/iman_project/Rabies__Weather__War_Combined_1.4.25.xlsx"
)

# ================== Preprocessing ==================
df = df.drop(columns=['Date', 'War Name', 'Event Per Year', 'Index Event ID'])
df['War in Israel'] = df['War in Israel'].map({'Yes': 1, 'No': 0})

# Convert month number to month name
month_names = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
df['Month'] = df['Month'].map(month_names)

# Features vs Targets
label_cols = ['Animal Species', 'Rabies Species', 'Settlement', 'Region_Weather']
target_cols = ['Region', 'Month']
num_cols = ['x', 'y', 'Avg Temperature', 'Monthly Precipitation (mm)', 'Rainy Days']

# --- OneHotEncoder for features ---
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(df[label_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(label_cols), index=df.index)
df = pd.concat([df.drop(columns=label_cols), encoded_df], axis=1)

# --- LabelEncoder for targets ---
target_encoders = {}
for col in target_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    target_encoders[col] = le

# --- StandardScaler for numeric features ---
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save preprocessors
joblib.dump(ohe, "preprocessing_onehot_encoder.pkl")
joblib.dump(target_encoders, "preprocessing_target_encoders.pkl")
joblib.dump(scaler, "preprocessing_scaler.pkl")

# Final X, y
X = df.drop(columns=target_cols)
y = df[target_cols]


# ================== Base models ==================
base_models = {
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),  # אין n_jobs בגרסה הזו
}

models = {name: MultiOutputClassifier(model) for name, model in base_models.items()}


# ================== Run evaluations ==================
results_df = evaluate_multioutput_models(X, y, models, n_runs=5)
catboost_df = evaluate_catboost_multi_target(X, y)

# Combine all results
final_results = pd.concat([results_df, catboost_df], ignore_index=True)


# ================== Final Output ==================
print("\nFinal Evaluation Results:")
print(final_results)

# Save best model
GB_model_IMAN = models['Gradient Boosting']
joblib.dump(GB_model_IMAN, "final_model_gradient_boosting.pkl")
print("✅ המודל נשמר כ- final_model_gradient_boosting.pkl")
