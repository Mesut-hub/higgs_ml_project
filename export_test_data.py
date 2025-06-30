import numpy as np
import joblib
from src.preprocessing import preprocess_data
from pathlib import Path  # Explicit import to avoid conflict

def export_test_data():
    # Load and preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data()
    
    # Split data (using same random state as before)
    from sklearn.model_selection import train_test_split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create output directory
    output_dir = Path('./outputs/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save test data
    print("Saving test data...")
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_test.npy', y_test)
    
    print(f"Test data exported successfully to {output_dir}!")

if __name__ == "__main__":
    export_test_data()