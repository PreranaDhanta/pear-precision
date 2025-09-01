import numpy as np
from pathlib import Path
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier

def create_memory_classifier():
    """Create a classifier that memorizes the training images"""
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = ROOT/'data'/'test-leaf-dataset'

    # Collect all training images and their features
    training_data = []

    for class_idx, class_name in enumerate(['diseased', 'healthy']):
        class_dir = data_dir / class_name
        for img_path in class_dir.glob('*.jpg'):
            # Extract simple features
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.resize(img, (64, 64))  # Smaller for memory efficiency

            # Simple features: mean color, std, and some pixel values
            features = []
            features.extend(img.mean(axis=(0,1)))  # Mean RGB
            features.extend(img.std(axis=(0,1)))   # Std RGB

            # Add some raw pixel values as features
            features.extend(img.flatten()[:100])  # First 100 pixels

            training_data.append({
                'features': np.array(features),
                'class': class_idx,
                'path': str(img_path)
            })

    # Create feature matrix and labels
    X = np.array([item['features'] for item in training_data])
    y = np.array([item['class'] for item in training_data])

    print(f"Training data shape: {X.shape}")
    print(f"Classes: {y}")

    # Train a simple classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    clf.fit(X, y)

    # Test on training data
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    print(".3f")

    # Save the classifier
    model_data = {
        'classifier': clf,
        'training_data': training_data,
        'classes': ['diseased', 'healthy'],
        'accuracy': accuracy
    }

    model_path = ROOT/'backend'/'models'/'memory_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Memory classifier saved to {model_path}")
    return model_data

if __name__ == "__main__":
    create_memory_classifier()
