# Multimodal Learning for Calorie Estimation

A deep learning model for estimating lunch calories using multiple data sources including CGM sequences, demographic information, microbiome profiles, and meal images. The model achieves a 37.73% improvement over baseline performance through multimodal integration.

## **Problem Statement**

The project aims to build a multimodal model to estimate lunch calories using diverse data sources collected from a nutrition research study:
- Continuous Glucose Monitoring (CGM) sequences
- Demographic and clinical information
- Microbiome profiles (Viome data)
- Meal images (breakfast and lunch)

## **Solution Architecture**

The model combines three specialized neural networks:
- Time Series Transformer for CGM data processing
- Text Transformer for demographic features
- Vision Transformer (ViT) with Food101 pretrained weights for meal images
- Multimodal integration through feature concatenation


## **Directory**

**config/**
- `config.yaml`: Contains model hyperparameters and training configurations

**data/**
- `dataset.py`: Implementation of MultiModalDataset class for data loading
- `preprocessing.py`: Functions for preprocessing CGM, demographic and image data

**models/**
- `time_series.py`: Time Series Transformer for CGM data
- `text.py`: Text Transformer for demographic features
- `vision.py`: Vision Transformer (ViT) for image processing
- `resnet.py`: ResNet Model for image processing
- `multimodal_v.py`: Combined model architecture (with Vit for images)
- `multimodal_r.py`: Combined model architecture (with ResNet for images)

**utils/**
- `losses.py`: RMSRE loss implementation
- `image_utils.py`: Image preprocessing and transformation utilities
- `time_utils.py`: Time series data processing functions

**Root Directory**
- `train.py`: Main training script
- `predict.py`: Inference script for generating predictions
- `requirements.txt`: Project dependencies


## **Performance**

- Final test RMSRE: 0.3274
- Baseline RMSRE: 0.5258
- Improvement: 37.73%
- Training RMSRE improved from 0.9910 to 0.3292
- Validation RMSRE improved from 0.9740 to 0.3196

## Installation

```bash
git clone https://github.com/ajayjagan2511/Multimodal-Learning-for-Calorie-Estimation.git
cd nutrition-calorie-estimation
pip install -r requirements.txt
```

## **Usage**

### Configuration and Training
1. Configure model parameters in `config/config.yaml`
2. Train the model:
```bash
python train.py
```
3. Generate predictions:
```bash
python predict.py
```

## **Requirements**

- Python 3.8+
- PyTorch 1.9+
- CUDA compatible GPU (recommended)
- Other dependencies:
```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
pyyaml>=5.4.0
google-colab
transformers>=4.5.0
pillow>=8.0.0
seaborn>=0.11.0
```

## **Training Files**
- `cgm_train.csv`          : CGM sequences and meal times
- `demo_viome_train.csv`   : Demographics and microbiome data
- `img_train.csv`          : Image data and fiber content
- `label_train.csv`        : Calorie labels

## **Testing Files**  
- `cgm_test.csv`          : CGM sequences and meal times
- `demo_viome_test.csv`   : Demographics and microbiome data
- `img_test.csv`          : Image data and fiber content

## **Contributing**

Contributions are welcome! Here's how you can get involved:

1. **Report Issues**:
   - Found a bug? Have a feature request? Open an issue in the GitHub repository.

2. **Suggest Enhancements**:
   - Propose ideas to improve the algorithm or its implementation.

3. **Submit Pull Requests**:
   - Fork the repository.
   - Make your changes in a new branch.
   - Open a pull request for review.

Please ensure all contributions adhere to the repository's coding standards and include sufficient documentation.


## **License**

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute the code, provided you include the original license. See the `LICENSE` file for details.


## **Acknowledgments**

Special thanks to:
- **Prof. Bobak Mortazavi**: For guidance, feedback, and inspiration during the project.





