# Titanic Survival Predictor
Titanic Survival Predictor for Kaggle's Titanic Survival Competition

## Implementation
Uses PyTorch and Deep NN to solve the problem, iterating on the Random Forest Approach. Model Architecture:

```py
nn.Linear(input_dim, 5),
nn.ReLU(),
nn.Dropout(0.15),
nn.Linear(5, 4),
nn.ReLU(),
nn.Dropout(0.25),
nn.Linear(4, 2),
```

## Usage
`python3 titanic.py` after installing dependencies; Config Variables are in the code
