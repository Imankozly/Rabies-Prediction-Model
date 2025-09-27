import pandas as pd

data = pd.read_excel('/Users/shryqb/PycharmProjects/PythonProject/bachlor/some_running/iman_project/Rabies__Weather__War_Combined_1.4.25.xlsx')

df = pd.DataFrame(data)
df = df.drop(columns='Date')
print(df.columns)

new_record = pd.DataFrame([{
    "Index Event ID": 100,  # כי עוד לא קרה בפועל
    "Event Per Year": 20,
    "Year": 2025,
    "Month": 9,
    "Animal Species": "Dog",
    "Rabies Species": "Na",
    "Region": "Amakim",
    "Settlement": "Yokneam",
    "x": 35.103,
    "y": 32.661,
    "Region_Weather": "North",
    "Avg Temperature": 28.5,
    "Monthly Precipitation (mm)": 5.2,
    "Rainy Days": 1,
    "War in Israel": "Yes",
    "War Name": "Iron Swords War"
}])


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

df_sorted = compute_similarity(df, new_record,df.columns)
#print(df_sorted.iloc[0])

print(df['Month'].unique())
