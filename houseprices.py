import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

DATA_PATH_TRAIN = 'data/train.csv'
DATA_PATH_TEST = 'data/test.csv'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5000
DEVICE = torch.device("cpu")

train_df = pd.read_csv(DATA_PATH_TRAIN)
test_df = pd.read_csv(DATA_PATH_TEST)

train_prices = train_df['SalePrice']
test_ids = test_df['Id']
train_features_df = train_df.drop(columns=['Id', 'SalePrice'])
test_features_df = test_df.drop(columns=['Id'])

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

train_features = preprocess_data(train_features_df)
test_features = preprocess_data(test_features_df, train_columns=train_features.columns)


X_train, X_val, y_train, y_val = train_test_split(train_features, train_prices, test_size=0.2, random_state=42)

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


class HousePriceDataset(Dataset):
    def __init__(self, features, prices=None):
        self.features = torch.tensor(features.values.astype(np.float32), dtype=torch.float32)
        self.prices = torch.tensor(prices.values, dtype=torch.float32).view(-1, 1) if prices is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.prices is not None:
            return self.features[idx], self.prices[idx]
        else:
            return self.features[idx]

train_dataset = HousePriceDataset(X_train, y_train)
val_dataset = HousePriceDataset(X_val, y_val)
test_dataset = HousePriceDataset(test_features)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        print(input_dim)
        super(HousePriceModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

input_dim = X_train.shape[1]
model = HousePriceModel(input_dim).to(DEVICE)

def rmse_loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets)**2))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6)

def train_model(model, train_loader, val_loader, optimizer, scheduler, price_scaler, epochs, device, patience=20):
    best_val_loss = float('inf')
    best_mae = float('inf')
    best_model_state = None
    no_improve_count = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []

        for features, prices in train_loader:
            features = features.to(device)
            prices = prices.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = rmse_loss(predictions, prices)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for features, prices in val_loader:
                features = features.to(device)
                prices = prices.to(device)

                predictions = model(features)
                loss = rmse_loss(predictions, prices)
                val_losses.append(loss.item())

                all_predictions.append(predictions)
                all_targets.append(prices)

        avg_val_loss = np.mean(val_losses)

        # Transform predictions back to original scale for MAE calculation
        predictions_tensor = torch.cat(all_predictions).cpu()
        targets_tensor = torch.cat(all_targets).cpu()

        predictions_orig = price_scaler.inverse_transform(predictions_tensor.numpy())
        targets_orig = price_scaler.inverse_transform(targets_tensor.numpy())

        val_mae = mean_absolute_error(targets_orig, predictions_orig)

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_mae = val_mae
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val MAE: ${val_mae:.2f}, LR: {optimizer.param_groups[0]['lr']:.6f}, Best Mae: ${best_mae:.2f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history

best, history = train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    price_scaler,
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
        test_predictions.extend(outputs.cpu().numpy())

test_predictions = price_scaler.inverse_transform(test_predictions).flatten()

submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': test_predictions})
print(submission_df)
submission_df.to_csv('submission.csv', index=False)