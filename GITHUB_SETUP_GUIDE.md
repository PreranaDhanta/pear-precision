# GitHub Setup Guide for Pear Precision Project

## Step 1: Create Repository on GitHub

1. Go to https://github.com/PreranaDhanta
2. Click the "+" icon in the top right and select "New repository"
3. Name your repository: `pear-precision`
4. Description: "Pear detection and disease classification using YOLOv8 and ResNet18"
5. Make it public or private as preferred
6. **DO NOT** initialize with README (we already have one)
7. Click "Create repository"

## Step 2: Push Your Code

After creating the repository, run these commands:

```bash
git remote add origin https://github.com/PreranaDhanta/pear-precision.git
git branch -M main
git push -u origin main
```

## Step 3: Upload to Google Colab

1. Go to https://colab.research.google.com/
2. Click "File" > "Upload notebook"
3. Upload the `pear_precision_colab.ipynb` file
4. Follow the instructions in the notebook to train your models

## Alternative: Clone from GitHub to Colab

If you prefer to clone directly from GitHub in Colab:

```python
!git clone https://github.com/PreranaDhanta/pear-precision.git
%cd pear-precision
```

## Important Notes

- Make sure to upload your `kaggle.json` file when prompted in Colab if you need to download datasets
- The training will be much faster on Colab's GPU compared to local CPU training
- Your trained models will be saved in the `runs/` directory

## Estimated Training Times

- **YOLOv8 training**: ~30-45 minutes on Colab GPU
- **ResNet18 training**: ~15-30 minutes on Colab GPU
- **Local CPU training**: 2-3 hours for YOLOv8
