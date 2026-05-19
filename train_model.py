import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv("KDDTrain.csv")
test_df = pd.read_csv("KDDTest.csv")

# 2. Preprocessing
# Drop redundant columns
drop_cols = ['difficulty_level']
train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

# Separate Features and Target
X_train_raw = train_df.drop(columns=['attack_type'])
y_train_raw = train_df['attack_type']
X_test_raw = test_df.drop(columns=['attack_type'])
y_test_raw = test_df['attack_type']

# One-Hot Encoding (Ensure train and test have same columns)
combined_X = pd.concat([X_train_raw, X_test_raw], axis=0)
combined_X_encoded = pd.get_dummies(combined_X)
X_train = combined_X_encoded.iloc[:len(train_df)]
X_test = combined_X_encoded.iloc[len(train_df):]

# Label Encoding
le = LabelEncoder()
le.fit(pd.concat([y_train_raw, y_test_raw]))
y_train = le.transform(y_train_raw)
y_test = le.transform(y_test_raw)

# 3. Build Hybrid Model
print("Training Hybrid Model (XGBoost + LightGBM)...")
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='mlogloss')
lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, verbose=-1)

hybrid_model = VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm)], voting='soft')
hybrid_model.fit(X_train, y_train)

# 4. Verify & Save
y_pred = hybrid_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Training Complete. Accuracy: {acc:.4f}")

# Save the model AND the metadata needed for the frontend
artifacts = {
    'model': hybrid_model,
    'features': combined_X_encoded.columns.tolist(),
    'label_encoder': le
}

joblib.dump(artifacts, 'hybrid_artifacts.joblib')
print("Success: 'hybrid_artifacts.joblib' has been created!")