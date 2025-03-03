import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

DATA_PATH_TRAIN = 'data/train.csv'
DATA_PATH_TEST = 'data/test.csv'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5000
DEVICE = torch.device("cpu")

train_df = pd.read_csv(DATA_PATH_TRAIN)
test_df = pd.read_csv(DATA_PATH_TEST)

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


X_train, X_val, y_train, y_val = train_test_split(train_features, survived, test_size=0.2, random_state=42)

feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_val = feature_scaler.transform(X_val)
test_features = feature_scaler.transform(test_features)

X_train = pd.DataFrame(X_train, columns=train_features.columns)
X_val = pd.DataFrame(X_val, columns=train_features.columns)
test_features = pd.DataFrame(test_features, columns=train_features.columns)


class TitanicDataset(Dataset):
    def __init__(self, features, prices=None):
        self.features = torch.tensor(features.values.astype(np.float32), dtype=torch.float32)
        self.survived = torch.tensor(prices.values, dtype=torch.long).view(-1, 1) if prices is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.survived is not None:
            return self.features[idx], self.survived[idx]
        else:
            return self.features[idx]

train_dataset = TitanicDataset(X_train, y_train)
val_dataset = TitanicDataset(X_val, y_val)
test_dataset = TitanicDataset(test_features)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 5),  # Match input to hidden layer
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(5, 4),  # No mismatch
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(4, 2),  # Match previous layer's output
        )

    def forward(self, x):
        return self.layers(x)

input_dim = X_train.shape[1]
model = TitanicModel(input_dim).to(DEVICE)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6)
criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device, patience=20):
    best_val_loss = float('inf')
    best_acc = float('inf')
    best_model_state = None
    no_improve_count = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        for features, survived in train_loader:
            features = features.to(device)
            survived = survived.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions.squeeze(), survived)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []
        accuracy = []

        with torch.no_grad():
            for features, survived in val_loader:
                features = features.to(device)
                survived = survived.to(device)

                predictions = model(features)
                loss = criterion(predictions.squeeze(), survived)
                val_losses.append(loss.item())

                accuracy.extend((torch.argmax(predictions, dim=1) == survived).cpu().numpy())

            acc = np.mean(accuracy)

        avg_val_loss = np.mean(val_losses)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_acc = acc
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: ${acc:.2f}, LR: {optimizer.param_groups[0]['lr']:.6f}, Best Mae: ${best_acc:.2f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

best = train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    EPOCHS,
    DEVICE,
    patience=200
)

best.eval()
test_predictions = []
with torch.no_grad():
    for features in test_loader:
        features = features.to(DEVICE)
        outputs = best(features)
        outputs = torch.argmax(outputs, dim=1).cpu().numpy()
        test_predictions.extend(outputs)

submission_df = pd.DataFrame({'Id': test_ids, 'Survived': test_predictions})
print(submission_df)
submission_df.to_csv('submission.csv', index=False)