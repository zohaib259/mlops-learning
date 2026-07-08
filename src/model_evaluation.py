import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 1. Load the trained model and test features
clf = pickle.load(open('model.pkl', 'rb'))
test_data = pd.read_csv('./data/features/test_bow.csv')

# 2. Separate features (X) and target labels (y)
X_test = test_data.iloc[:, 0:-1].values
y_test_raw = test_data.iloc[:, -1].values

# 3. Force convert y_test to strings and map to 1 and 0
y_test_str = np.array(y_test_raw, dtype=str)
y_test = np.where(y_test_str == 'happiness', 1, 0)

# 4. Generate model predictions
y_pred_raw = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# 5. Force convert y_pred to strings and map to 1 and 0
y_pred_str = np.array(y_pred_raw, dtype=str)
y_pred = np.where(y_pred_str == 'happiness', 1, 0)

# 6. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

# 7. Package metrics into a dictionary
metrics_dict = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'auc': float(auc)
}

# 8. Save metrics to a JSON file
with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)

print("🎉 Success! Both targets are perfectly aligned as binary numbers.")
print("Results saved to 'metrics.json'.")