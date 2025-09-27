import joblib

# טען את המודל
model = joblib.load("DB_model_IMAN.pkl")

# הדפס את שמות התכונות (feature names)
print(model.feature_names_in_)

