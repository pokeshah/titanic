import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


n=10

# Load data (using the same preprocessing as the PyTorch example)
DATA_PATH_TRAIN = 'data/train.csv'
DATA_PATH_TEST = 'data/test.csv'  # Keep test data for final evaluation if needed.

train_df = pd.read_csv(DATA_PATH_TRAIN)
test_df = pd.read_csv(DATA_PATH_TEST) # Load test data

survived = train_df['Survived']
test_ids = test_df['PassengerId']
train_df = train_df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])
test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

def preprocess_data(data, train_columns=None):
    data.dropna(axis=1, thresh=int(0.85 * len(data)), inplace=True)

    numeric_columns = data.select_dtypes(exclude=['object']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    data[categorical_columns] = data[categorical_columns].fillna('Unknown')

    data = pd.get_dummies(data, columns=categorical_columns, dummy_na=False)

    if train_columns is not None:
        missing_cols = set(train_columns) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        data = data[train_columns]

    return data

train_features = preprocess_data(train_df)
test_features = preprocess_data(test_df, train_columns=train_features.columns)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, survived, test_size=0.2, random_state=42)

feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_val = feature_scaler.transform(X_val)
test_features = feature_scaler.transform(test_features)

price_scaler = StandardScaler()
y_train = price_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val = price_scaler.transform(y_val.values.reshape(-1, 1)).flatten()

X_train = pd.DataFrame(X_train, columns=train_features.columns)
X_val = pd.DataFrame(X_val, columns=train_features.columns)
test_features = pd.DataFrame(test_features, columns=train_features.columns)
y_train = pd.Series(y_train)
y_val = pd.Series(y_val)

# --- RandomForestRegressor ---

# 1. Basic RandomForestRegressor (with default parameters)
rf_basic = RandomForestRegressor(random_state=42, n_jobs=-1)  # n_jobs=-1 uses all available cores
rf_basic.fit(X_train, y_train)

y_pred_basic = rf_basic.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_basic)
classification_rep = classification_report(y_val, y_pred_basic)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

# 2. Feature Importances
importances = rf_basic.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

print(f"\nTop {n} Feature Importances:")
print(feature_importance_df.head(n))

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_importance_df['Feature'][:n], feature_importance_df['Importance'][:n])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(f'Top {n} Feature Importances (Basic RandomForest)')
plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
plt.tight_layout()
plt.show()

print(len(feature_importance_df))
print(sum(feature_importance_df['Importance'][:n]))
# 3. Train on Top 30 Features
top_30_features = feature_importance_df['Feature'][:n].tolist()
X_train_top30 = X_train[top_30_features]
X_val_top30 = X_val[top_30_features]

rf_top30 = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_top30.fit(X_train_top30, y_train)

y_pred_top30 = rf_top30.predict(X_val_top30)
mae_top30 = mean_absolute_error(y_val, y_pred_top30)
rmse_top30 = np.sqrt(mean_squared_error(y_val, y_pred_top30))

print(f"\nRandomForest (Top {n} Features) - MAE: {mae_top30:.2f}, RMSE: {rmse_top30:.2f}")