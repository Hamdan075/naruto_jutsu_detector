import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def normalize_row(row):
    """
    Normalizes a row of 126 coordinate features (2 hands * 21 landmarks * 3 coords).
    It centers the hands relative to their collective center of mass and scales them.
    This ensures that the jutsu works whether you stand on the left or right side of the camera!
    """
    coords = row.values.reshape(-1, 3)
    
    # Filter out padded zeros (for missing hands)
    valid_mask = np.any(coords != 0, axis=1)
    
    if not np.any(valid_mask):
        return row
        
    valid_coords = coords[valid_mask]
    centroid = np.mean(valid_coords, axis=0)
    
    translated_coords = np.copy(coords)
    translated_coords[valid_mask] -= centroid
    
    distances = np.linalg.norm(translated_coords[valid_mask], axis=1)
    max_dist = np.max(distances)
    
    if max_dist > 0:
        translated_coords[valid_mask] /= max_dist
        
    return pd.Series(translated_coords.flatten(), index=row.index)

def main():
    print("================================")
    print(" Jutsu Model Trainer ")
    print("================================")
    
    if not os.path.exists('gestures.csv'):
        print("Error: 'gestures.csv' not found.")
        print("Please run 'python collect_data.py' first to record some gestures!")
        return

    print("Loading dataset 'gestures.csv'...")
    df = pd.read_csv('gestures.csv')
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"Loaded {len(df)} samples.")
    print(f"Found {len(y.unique())} distinct gestures: {', '.join(y.unique())}")
    
    print("\nNormalizing coordinates for position-independence...")
    X_norm = X.apply(normalize_row, axis=1)
    
    # Split into train and test sets to verify the model actually learned
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)
    
    print("\nTraining AI Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Model trained! Validation Accuracy: {acc * 100:.2f}%")
    
    # Save the trained model
    with open('gesture_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("\nSuccess! Saved trained model to 'gesture_model.pkl'.")
    print("We can now update detector.py to use this model!")

if __name__ == "__main__":
    main()
