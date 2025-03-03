import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

n = 6

# Load data (using the same preprocessing as the PyTorch example)
DATA_PATH_TRAIN = 'data/train.csv'
DATA_PATH_TEST = 'data/test.csv'  # Keep test data for final evaluation if needed.

train_df = pd.read_csv(DATA_PATH_TRAIN)
test_df = pd.read_csv(DATA_PATH_TEST)  # Load test data

survived = train_df['Survived']
test_ids = test_df['PassengerId']
train_df = train_df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'])
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

ga = np.array([])
while True:
    X_train, X_val, y_train, y_val = train_test_split(train_features, survived, test_size=0.2)

    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_val = feature_scaler.transform(X_val)
    test_features = feature_scaler.transform(test_features)

    X_train = pd.DataFrame(X_train, columns=train_features.columns)
    X_val = pd.DataFrame(X_val, columns=train_features.columns)
    test_features = pd.DataFrame(test_features, columns=train_features.columns)

    # --- RandomForestRegressor ---

    # 1. Basic RandomForestRegressor (with default parameters)
    rf_basic = RandomForestRegressor(n_jobs=-1)  # n_jobs=-1 uses all available cores
    rf_basic.fit(X_train, y_train)

    y_pred_basic = rf_basic.predict(X_val)

    # Convert regression predictions to binary (0 or 1) for classification metrics
    y_pred_binary = (y_pred_basic > 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred_binary)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}, ga = {np.mean(ga):.2f}")
    ga = np.append(ga, accuracy)

# # 2. Feature Importances
# importances = rf_basic.feature_importances_
# feature_names = X_train.columns
# feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
#
# print(f"\nTop {n} Feature Importances:")
# print(feature_importance_df.head(n))
#
# # Plot feature importances
# plt.figure(figsize=(12, 6))
# plt.barh(feature_importance_df['Feature'][:n], feature_importance_df['Importance'][:n])
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title(f'Top {n} Feature Importances (Basic RandomForest)')
# plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
# plt.tight_layout()
# plt.show()
#
# print(len(feature_importance_df))
# print(sum(feature_importance_df['Importance'][:n]))
#
# # 3. Train on Top n Features
# top_n_features = feature_importance_df['Feature'][:n].tolist()
# X_train_topn = X_train[top_n_features]
# X_val_topn = X_val[top_n_features]
#
# rf_topn = RandomForestRegressor(random_state=42, n_jobs=-1)
# rf_topn.fit(X_train_topn, y_train)
#
# y_pred_topn = rf_topn.predict(X_val_topn)
#
# y_pred_topn_binary = (y_pred_topn > 0.5).astype(int)
#
# accuracy_topn = accuracy_score(y_val, y_pred_topn_binary)
# classification_rep_topn = classification_report(y_val, y_pred_topn_binary)
#
# print(f"\nRandomForest (Top {n} Features) - Accuracy: {accuracy_topn:.2f}")
# print("\nClassification Report:\n", classification_rep_topn)