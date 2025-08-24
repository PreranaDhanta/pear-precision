# Pear Precision - Pear Detection and Disease Classification

A comprehensive computer vision project for detecting pears and classifying leaf diseases using YOLOv8 and ResNet18 models.

## Project Structure

```
pear-precision/
├── backend/           # FastAPI backend and model training scripts
├── data/             # Dataset files and configurations
├── frontend/         # Web interface
├── runs/            # Training results and model outputs
└── requirements.txt  # Python dependencies
```

## Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/pear-precision.git
cd pear-precision
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Training in Google Colab

1. Open the provided Colab notebook: `pear_precision_colab.ipynb`
2. Upload your Kaggle API key (if needed for dataset downloads)
3. Run all cells to train the models

## Models

- **YOLOv8**: Pear detection and localization
- **ResNet18**: Leaf disease classification

## Usage

1. Start the backend server:
```bash
python backend/app.py
```

2. Open `frontend/index.html` in a web browser

## Dataset

The project uses the pear640 dataset from Kaggle for object detection training.

## License

MIT License
