import torch
import torchvision as tv
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import cv2

def extract_features(image_path):
    """Extract simple features from image"""
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (256, 256))

    # Color histogram features
    hist_features = []
    for channel in range(3):  # RGB channels
        hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)

    # Texture features (simple statistics)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture_features = [
        gray.mean(),      # Mean intensity
        gray.std(),       # Standard deviation
        gray.var(),       # Variance
        cv2.Laplacian(gray, cv2.CV_64F).var()  # Laplacian variance (texture)
    ]

    # Shape features
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        shape_features = [area, perimeter, area/(perimeter+1)]  # Compactness
    else:
        shape_features = [0, 0, 0]

    return np.array(hist_features + texture_features + shape_features)

def create_simple_classifier():
    """Create a simple classifier using traditional ML"""
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = ROOT/'data'/'test-leaf-dataset'

    # Collect features and labels
    features = []
    labels = []

    for class_idx, class_name in enumerate(['diseased', 'healthy']):
        class_dir = data_dir / class_name
        for img_path in class_dir.glob('*.jpg'):
            try:
                feature_vector = extract_features(img_path)
                features.append(feature_vector)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    features = np.array(features)
    labels = np.array(labels)

    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    print(".3f")
    print(".3f")

    # Save the classifier
    model_data = {
        'classifier': clf,
        'classes': ['diseased', 'healthy'],
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    }

    model_path = ROOT/'backend'/'models'/'simple_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Simple classifier saved to {model_path}")
    return model_data

if __name__ == "__main__":
    create_simple_classifier()
