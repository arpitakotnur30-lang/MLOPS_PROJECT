import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def train():
    mlflow.set_experiment("housing_experiment")
    
    # Resolve path relative to this script's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "../data/housing.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Did you run dvc pull?")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    
    # Train-test split (handles small sample size safely)
    test_size = 0.4 if len(df) < 10 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    n_estimators = 50
    learning_rate = 0.1
    max_depth = 3
    
    with mlflow.start_run():
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Log to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Training complete! MSE: {mse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    train()