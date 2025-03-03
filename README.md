# House Price Predictor
House Price Predictor for Kaggle's House Prices: Advanced Regression Techniques competition

## Implementation
Uses PyTorch and Deep NN to solve the problem, iterating on the Random Forest Approach. Model Architecture:
```py
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
```

## Usage
`python3 houseprices.py` after installing dependencies; Config Variables are in the code
