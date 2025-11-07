import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load data exported from MATLAB
X = pd.read_csv('final_features.csv')
Y_table = pd.read_csv('final_labels.csv')
Y = Y_table['Label'] # Assuming the label column name is 'Label'

print(f"Data loaded successfully, containing {X.shape[0]} samples and {X.shape[1]} features.")

# 2. Initialize a Scikit-learn Random Forest model
#    (TreeBagger(100) corresponds to n_estimators=100)
#    oob_score=True corresponds to 'OOBPrediction', 'on'
model = RandomForestClassifier(n_estimators=100, 
                               oob_score=True, 
                               random_state=42) # random_state ensures reproducibility

# 3. Train the model
print("Starting model training...")
model.fit(X, Y)

print(f"Model training complete. OOB Accuracy: {model.oob_score_:.4f}")

# 4. (Optional) Check feature importance (for comparison with your MATLAB results)
# importances = pd.Series(model.feature_importances_, index=X.columns)
# print("Top 10 Features:")
# print(importances.nlargest(10))

# 5. Save this *Python native* model
model_filename = 'py_random_forest.joblib'
joblib.dump(model, model_filename)

print(f"Model saved to {model_filename}")