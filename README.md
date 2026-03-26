# Bone Fracture Classification using Deep Learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning project for classifying bone fracture X-ray images using TensorFlow/Keras with transfer learning and modern MLOps practices.

## 🦴 Project Overview

This project implements a complete end-to-end machine learning pipeline for bone fracture classification from X-ray images. The system uses 5 different neural network architectures and provides both programmatic and web-based interfaces for model training and inference.

### Key Features

- **Multi-Model Architecture**: 5 different neural networks (MobileNetV2, ResNet50, EfficientNetB0, DenseNet121, Custom CNN)
- **Transfer Learning**: Pre-trained models with ImageNet weights for better performance
- **Data Augmentation**: Real-time image augmentation during training
- **Model Evaluation**: Comprehensive metrics, confusion matrices, and performance comparison
- **Web Interface**: Streamlit-based application for easy model deployment and inference
- **Modular Design**: Clean, maintainable code structure following ML engineering best practices

## 📊 Dataset

The project uses a dataset containing 10 classes of bone fractures:

1. Avulsion fracture
2. Comminuted fracture
3. Fracture Dislocation
4. Greenstick fracture
5. Hairline Fracture
6. Impacted fracture
7. Longitudinal fracture
8. Oblique fracture
9. Pathological fracture
10. Spiral Fracture

### Dataset Structure
```
dataset/
├── Avulsion fracture/
│   ├── Train/
│   └── Test/
├── Comminuted fracture/
│   ├── Train/
│   └── Test/
└── ... (other fracture types)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.13+
- CUDA-compatible GPU (recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bone-fracture-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Place your dataset in the `dataset/` folder following the structure above
   - Ensure images are in JPG/PNG format

### Usage

#### Training Models

Run the Jupyter notebook for complete training and evaluation:

```bash
jupyter notebook notebooks/training.ipynb
```

Or run individual components:

```python
from src.data_loader import DataLoader
from src.model_builder import ModelBuilder
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator

# Load and preprocess data
loader = DataLoader()
train_ds, val_ds, class_names, num_classes = loader.load_dataset()

# Build models
builder = ModelBuilder()
models = builder.build_all_models(num_classes)

# Train models
trainer = ModelTrainer()
trained_models, histories = trainer.train_all_models(models, train_ds, val_ds)

# Evaluate models
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(trained_models, val_ds, class_names)
```

#### Web Application

Launch the Streamlit web application:

```bash
streamlit run app/app.py
```

The app provides:
- Image upload interface
- Real-time classification
- Model comparison visualizations
- Performance metrics dashboard

## 🏗️ Project Structure

```
bone-fracture-classification/
├── dataset/                          # Dataset folder
│   ├── Avulsion fracture/
│   ├── Comminuted fracture/
│   └── ... (other classes)
├── src/                             # Source code
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── model_builder.py            # Model architectures
│   ├── trainer.py                  # Training logic
│   ├── evaluator.py                # Model evaluation
│   └── utils.py                    # Utility functions
├── notebooks/                       # Jupyter notebooks
│   └── training.ipynb              # Main training notebook
├── app/                            # Web application
│   └── app.py                      # Streamlit app
├── outputs/                        # Generated outputs
│   ├── models/                     # Saved models
│   ├── plots/                      # Training plots
│   ├── metrics/                    # Evaluation metrics
│   └── logs/                       # Training logs
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```

## 🧠 Model Architectures

### 1. MobileNetV2
- Lightweight architecture optimized for mobile devices
- Depthwise separable convolutions
- Pre-trained on ImageNet

### 2. ResNet50
- 50-layer residual network
- Skip connections to combat vanishing gradients
- Excellent feature extraction capabilities

### 3. EfficientNetB0
- Compound scaling for optimal performance
- Balances network depth, width, and resolution
- State-of-the-art efficiency

### 4. DenseNet121
- Dense connectivity between layers
- Feature reuse through concatenation
- Parameter efficiency

### 5. Custom CNN
- Tailored architecture for fracture classification
- Convolutional blocks with batch normalization
- Global average pooling for classification

## 📈 Training Configuration

- **Image Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall

### Data Augmentation
- Random horizontal/vertical flips
- Random rotations (±20°)
- Random zoom (0.8-1.2x)
- Random brightness adjustments

## 🔍 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity of the model
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class-wise performance

## 🌐 Web Application Features

### Image Classification
- Drag-and-drop image upload
- Support for JPG, PNG, JPEG formats
- Real-time prediction with confidence scores

### Model Comparison
- Side-by-side model performance
- Interactive probability visualizations
- Best model recommendations

### Results Dashboard
- Training history plots
- Model architecture summaries
- Performance metrics overview

## 🛠️ Development

### Adding New Models

1. Extend the `ModelBuilder` class in `src/model_builder.py`
2. Add the new model method following the existing pattern
3. Update the `build_all_models()` method

### Custom Data Loading

Modify `src/data_loader.py` to support different dataset formats or additional preprocessing steps.

### Web App Customization

Edit `app/app.py` to add new features or modify the user interface.

## 📋 Requirements

### Hardware
- **Minimum**: CPU with 8GB RAM
- **Recommended**: GPU with 16GB+ VRAM (NVIDIA RTX 30-series or equivalent)

### Software
- Python 3.8+
- TensorFlow 2.13+
- CUDA 11.8+ (for GPU support)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the excellent deep learning framework
- Streamlit team for the amazing web app framework
- Medical imaging community for providing datasets and research

## 📞 Contact

For questions or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is for educational and research purposes. Always consult medical professionals for actual fracture diagnosis and treatment decisions.
3. ResNet50              | Accuracy: 0.9021 (90.21%)
4. Custom CNN            | Accuracy: 0.8756 (87.56%)
5. MobileNetV2           | Accuracy: 0.8534 (85.34%)

Best Model: EfficientNetB0
Best Accuracy: 0.9245 (92.45%)
```

## 🌐 Deployment with Streamlit

### Run the Web App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Features

- **Upload Custom Image**: Upload any X-ray image for classification
- **Sample Image**: Test with a sample from the dataset
- **Real-time Prediction**: Instant classification with confidence scores
- **Visualization**: 
  - Original image display
  - Confidence score bar chart
  - Predictions table
  - Top class highlight
- **Model Info**: Display current model and class information

### Usage Steps

1. Run: `streamlit run app.py`
2. Upload an X-ray image (JPG, PNG, BMP, GIF)
3. View predictions and confidence scores
4. See detailed breakdown of all class predictions

## 📊 Model Comparison

| Model | Speed | Accuracy* | Size | Parameters |
|-------|-------|-----------|------|------------|
| MobileNetV2 | ⚡⚡ (Fast) | ~85% | 9MB | 2.3M |
| ResNet50 | ⚡ (Medium) | ~90% | 98MB | 23M |
| EfficientNetB0 | ⚡⚡ (Fast) | ~92% | 29MB | 5.3M |
| DenseNet121 | ⚡ (Medium) | ~91% | 33MB | 7.1M |
| Custom CNN | ⚡ (Medium) | ~87% | 120MB | 26M |

*Approximate values; actual accuracy depends on your specific dataset

## 🔧 Configuration

To modify training parameters, edit the configuration section in `train.py`:

```python
IMG_SIZE = 224          # Image size (keep at 224 for transfer learning)
BATCH_SIZE = 32         # Batch size for training
EPOCHS = 10             # Number of training epochs
DATASET_PATH = "Bone Break Classification"  # Path to dataset
MODELS_PATH = "models"  # Where to save models
```

## ⚠️ Troubleshooting

### Issue: "Dataset path not found"
**Solution**: Ensure the `Bone Break Classification` folder is in the project root directory alongside Python scripts.

### Issue: Out of Memory Error
**Solution**: 
- Reduce BATCH_SIZE in train.py (try 16 or 8)
- Close other applications
- Use GPU acceleration

### Issue: No models found during evaluation
**Solution**: Run `train.py` first to create models before running `evaluate.py`

### Issue: Streamlit app shows "No best model found"
**Solution**: Run `evaluate.py` to generate the `best_model.txt` file

### Issue: Slow training
**Solution**:
- Use GPU: Install CUDA and cuDNN
- Reduce EPOCHS
- Start with smaller models (MobileNetV2)

## 📝 Project Files Description

### `train.py`
- Loads dataset with 80/20 train-val split
- Implements 5 different model architectures
- Applies data augmentation
- Trains all models with early stopping
- Saves models in Keras format

### `evaluate.py`
- Loads all trained models
- Evaluates on validation dataset
- Ranks models by accuracy
- Saves best model name to file

### `app.py`
- Streamlit web interface
- Image upload and preprocessing
- Model inference
- Visualization of predictions
- Class information display

### `requirements.txt`
- TensorFlow 2.13+
- Streamlit 1.28+
- NumPy, Matplotlib, Scikit-learn
- Pillow for image processing

## 🎯 Performance Tips

1. **Start with MobileNetV2** - Fast training, good for testing
2. **Verify with EfficientNetB0** - Best accuracy-speed balance
3. **Use GPU** - Dramatically speeds up training
4. **Increase EPOCHS** - Better accuracy with 20-50 epochs
5. **More Data** - More training samples improve generalization

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Applications](https://keras.io/api/applications/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Transfer Learning: https://www.tensorflow.org/tutorials/images/transfer_learning

## 📄 License

This project is provided for educational purposes.

## 🤝 Contributing

Feel free to:
- Add more fracture types
- Implement additional models
- Improve the Streamlit UI
- Optimize training pipeline

## ⚠️ Disclaimer

**This is an educational project.** The model predictions should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify dataset structure
3. Ensure all dependencies are installed correctly
4. Check Python version (3.8+)

---

**Last Updated**: March 2026  
**TensorFlow Version**: 2.13+  
**Python Version**: 3.8+
