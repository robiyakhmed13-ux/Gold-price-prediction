# =============================================================================
# Gold Price Prediction using Random Forest Regressor
# Author: [Your Name]
# Dataset: https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the gold price dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


# =============================================================================
# 2. EDA & Correlation
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """Correlation heatmap, GLD price distribution, and GLD over time."""
    gold_data = df.drop(columns='Date')

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        gold_data.corr(), cbar=True, square=True,
        fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues'
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Heatmap saved as 'correlation_heatmap.png'")

    # Gold price distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(df['GLD'], kde=True, color='gold')
    plt.title("Gold (GLD) Price Distribution")
    plt.xlabel("GLD Price")
    plt.tight_layout()
    plt.savefig("gld_distribution.png", dpi=150)
    plt.show()
    print("Distribution plot saved as 'gld_distribution.png'")

    # GLD price over time
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(df['Date']), df['GLD'], color='goldenrod', linewidth=1.2)
    plt.title("Gold (GLD) Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("GLD Price (USD)")
    plt.tight_layout()
    plt.savefig("gld_over_time.png", dpi=150)
    plt.show()
    print("Time series plot saved as 'gld_over_time.png'")

    print(f"\nCorrelation with GLD:\n{gold_data.corr()['GLD'].sort_values(ascending=False)}")


# =============================================================================
# 3. Feature / Target Split
# =============================================================================

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=['Date', 'GLD'], axis=1)
    Y = df['GLD']
    print(f"Features: {X.shape} | Target: {Y.shape}")
    return X, Y


# =============================================================================
# 4. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.2, random_state=2):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 5. Model Training
# =============================================================================

def train_model(X_train, Y_train):
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)
    print("Model training complete.")
    return model


# =============================================================================
# 6. Model Evaluation
# =============================================================================

def evaluate_model(model, X_train, Y_train, X_test, Y_test) -> None:
    """R² score, MAE, and actual vs predicted line plot."""
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    print(f"\nTraining R²  : {r2_score(Y_train, train_preds):.4f}")
    print(f"Test     R²  : {r2_score(Y_test,  test_preds):.4f}")
    print(f"Test     MAE : {mean_absolute_error(Y_test, test_preds):.4f}")

    # Actual vs Predicted line plot
    plt.figure(figsize=(12, 5))
    plt.plot(Y_test.values, color='blue', label='Actual GLD Price', linewidth=1.2)
    plt.plot(test_preds,    color='green', label='Predicted GLD Price',
             linewidth=1.2, linestyle='--')
    plt.title("Actual vs Predicted Gold (GLD) Price")
    plt.xlabel("Test Sample Index")
    plt.ylabel("GLD Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.show()
    print("Plot saved as 'actual_vs_predicted.png'")

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances.sort_values(ascending=True).plot(
        kind='barh', figsize=(7, 5), color='goldenrod',
        title='Feature Importances (Random Forest)'
    )
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=150)
    plt.show()
    print("Feature importances saved as 'feature_importances.png'")


# =============================================================================
# 7. Predictive System
# =============================================================================

def predict_gold_price(model, input_data: tuple) -> None:
    """
    Predict the GLD ETF price for given market indicators.

    Parameters
    ----------
    input_data : tuple
        Values for all features except Date and GLD, in column order:
        (SPX, USO, SLV, EUR/USD)
    """
    arr = np.asarray(input_data).reshape(1, -1)
    price = model.predict(arr)[0]
    print(f"\n🥇 Predicted GLD Price: ${price:.2f}")


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "gld_price_data.csv"   # update path if needed

    df = load_data(DATA_PATH)
    print("\nFirst 5 rows:\n", df.head())

    plot_eda(df)

    X, Y = split_features_target(df)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    model = train_model(X_train, Y_train)
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # Sample prediction (SPX, USO, SLV, EUR/USD)
    sample = (1500.0, 35.0, 15.0, 1.25)
    predict_gold_price(model, sample)
