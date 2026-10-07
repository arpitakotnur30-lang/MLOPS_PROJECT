import os
import pandas as pd
from src.train import train

def test_data_existence():
    """Verify that the housing dataset exists locally."""
    assert os.path.exists("data/housing.csv"), "housing.csv is missing!"

def test_data_columns():
    """Verify that the dataset contains expected columns."""
    df = pd.read_csv("data/housing.csv")
    expected_columns = {"MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"}
    assert expected_columns.issubset(df.columns), "Dataset columns do not match expected schema!"

def test_training_pipeline():
    """Verify that the training function runs without crashing."""
    try:
        train()
    except Exception as e:
        assert False, f"Training script failed with error: {e}"