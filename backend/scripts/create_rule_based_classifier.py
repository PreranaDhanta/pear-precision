import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import pickle

def analyze_leaf_health(image_path):
    """Analyze leaf health using simple rules"""
    img = cv2.imread(str(image_path))
    if img is None:
        return {"healthy": 0.5, "diseased": 0.5}

    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Healthy leaves typically have:
    # 1. More green color (higher green channel values)
    # 2. Less brown/yellow spots (lower red channel values)
    # 3. More uniform color distribution

    # Extract color channels
    b, g, r = cv2.split(img)

    # Calculate green intensity (healthy indicator)
    green_intensity = np.mean(g)

    # Calculate brown/yellow spots (disease indicator)
    # Brown spots appear as areas with high red and low green
    brown_mask = (r > 150) & (g < 100) & (b < 100)
    brown_percentage = np.sum(brown_mask) / brown_mask.size

    # Calculate color uniformity (healthy leaves are more uniform)
    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)
    color_uniformity = 1 / (1 + (std_r + std_g + std_b) / 300)  # Normalize to 0-1

    # Calculate texture roughness (diseased leaves are often rougher)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Simple rule-based classification
    # Healthy score based on green intensity and uniformity
    healthy_score = (green_intensity / 255 * 0.6) + (color_uniformity * 0.4)

    # Diseased score based on brown spots and texture roughness
    diseased_score = (brown_percentage * 0.7) + (min(laplacian_var / 500, 1) * 0.3)

    # Normalize scores
    total = healthy_score + diseased_score
    if total > 0:
        healthy_prob = healthy_score / total
        diseased_prob = diseased_score / total
    else:
        healthy_prob = 0.5
        diseased_prob = 0.5

    return {
        "healthy": float(healthy_prob),
        "diseased": float(diseased_prob)
    }

def create_rule_based_classifier():
    """Create a rule-based classifier for leaf health"""
    ROOT = Path(__file__).resolve().parents[2]
    data_dir = ROOT/'data'/'test-leaf-dataset'

    # Test the classifier on training data
    results = []
    for class_name in ['healthy', 'diseased']:
        class_dir = data_dir / class_name
        for img_path in class_dir.glob('*.jpg'):
            analysis = analyze_leaf_health(img_path)
            results.append({
                'path': str(img_path),
                'true_class': class_name,
                'prediction': 'healthy' if analysis['healthy'] > analysis['diseased'] else 'diseased',
                'healthy_prob': analysis['healthy'],
                'diseased_prob': analysis['diseased']
            })

    # Calculate accuracy
    correct = sum(1 for r in results if r['true_class'] == r['prediction'])
    accuracy = correct / len(results) if results else 0

    print(f"Rule-based classifier accuracy: {accuracy:.3f}")
    print("Sample predictions:")
    for r in results[:4]:  # Show first 4 results
        print(f"  {r['path'].split('/')[-1]}: True={r['true_class']}, Pred={r['prediction']}, "
              ".3f")

    # Save the rule-based classifier (just the function reference)
    model_data = {
        'classifier_type': 'rule_based',
        'accuracy': accuracy,
        'classes': ['diseased', 'healthy']
    }

    model_path = ROOT/'backend'/'models'/'rule_based_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Rule-based classifier saved to {model_path}")
    return model_data

if __name__ == "__main__":
    create_rule_based_classifier()
