# Rabies Prediction Model

This project analyzes and predicts confirmed rabies cases in Israel, based on real data collected from 2006 until today.  
The workflow included **data preprocessing, cleaning, feature engineering, and machine learning modeling**.  
The main model used is **Gradient Boosting**, designed to detect patterns and forecast possible outbreaks.

## Goals
- Collect and clean real rabies case data.
- Standardize geographic and species information.
- Build predictive models for outbreak detection.
- Provide insights for public health monitoring and decision-making.

## Methodology
1. **Data Preprocessing**
   - Removal of irrelevant columns and duplicates.
   - Standardization of settlement names and translations.
   - Handling missing values with consistent logic.
   - Adding new features (e.g., month, location coordinates).

2. **Modeling**
   - Gradient Boosting for prediction of outbreak likelihood.
   - Evaluation and validation of model performance.

3. **Monitoring & Maintenance**
   - Tracking key factors like geography, seasonality, and animal species.
   - Detecting when the model becomes outdated.

## Technologies
- Python: pandas, numpy, scikit-learn, matplotlib, seaborn


## Results
The model shows promising performance in forecasting rabies outbreaks and offers valuable insights for improving monitoring and prevention strategies.


