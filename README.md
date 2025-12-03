# Deepfake Audio Detection

A machine learning project for detecting deepfake (AI-synthesized) audio vs. real audio using deep learning models and pre-trained transformers.
This project implements multiple approaches to classify audio as either fake (AI-generated) or real (authentic). It leverages both custom CNN-based models and state-of-the-art pre-trained transformer models like Wav2Vec2.

## Features

- **Multiple Model Architectures**:
  - Custom CNN-based audio classifier for baseline performance
  - Wav2Vec2 feature extraction with custom classification head
  - Pre-trained audio classification models from HuggingFace transformers
  
- **Audio Processing**:
  - Mel spectrogram visualization
  - Waveform normalization and padding
  - Automatic sample rate conversion to 16kHz
  
- **Training & Evaluation**:
  - Train/validation/test split with stratification
  - Class-weighted loss for balanced training
  - Confusion matrices and classification reports
  - Support for multiple hyperparameter configurations
  
- **Dataset Support**:
  - Uses the "Fake or Real Dataset" from Kaggle
  - Automatic data downloading and extraction
  - Supports training on subsets of data (useful for experimentation)

## Requirements

- Python 3.7+
- PyTorch with CUDA support (optional but recommended)
- torchaudio
- transformers
- librosa
- scikit-learn
- matplotlib
- pandas
- numpy
- Streamlit (for the web app)
- FFmpeg

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arshad-perampalli/Deepfake_Audio_Detection.git
cd Deepfake_Audio_Detection
```

2. Install dependencies:
```bash
pip install torch torchaudio transformers librosa scikit-learn matplotlib pandas numpy streamlit
apt-get install ffmpeg
```

3. Set up Kaggle API credentials:
   - Download your Kaggle API key from https://www.kaggle.com/settings/account
   - Place `kaggle.json` in `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Dataset

The project uses the **Fake or Real Dataset** from Kaggle, which contains:
- Training, validation, and testing subsets
- Fake (AI-synthesized) and real audio samples
- WAV format audio files

The dataset is automatically downloaded and extracted using the Kaggle API.

## Usage

### Running the Jupyter Notebook

Open `Deepfake_Audiou_Detection.ipynb` in a Jupyter environment (Google Colab recommended):

```bash
jupyter notebook Deepfake_Audiou_Detection.ipynb
```

The notebook includes:
1. Data loading and exploration
2. Audio visualization and spectrogram analysis
3. Custom CNN model training
4. Wav2Vec2 feature extractor model training
5. Pre-trained transformer model fine-tuning
6. Model evaluation with confusion matrices and classification reports
7. Model saving and deployment

### Key Sections

1. **Data Preparation**: Downloads and prepares the dataset, creates DataFrames with train/val/test splits
2. **Model Architecture**: Implements `AudioClassifier` CNN and `Wav2Vec2Classifier` wrapper
3. **Training**: Trains models with configurable hyperparameters and class weighting
4. **Evaluation**: Evaluates on test set and generates performance metrics
5. **Deployment**: Optionally saves model to Google Drive or creates Streamlit web app

## Methodology
![Image](https://github.com/arshad-perampalli/Deepfake_Audio_Detection/blob/main/Methodology.png?raw=true)

## Training Configuration

Default hyperparameters:
- **Batch Size**: 32-64
- **Learning Rate**: 1e-3 (custom model), 3e-5 (Wav2Vec2)
- **Optimizer**: Adam with weight decay (1e-4)
- **Loss Function**: Cross-entropy with class weighting
- **Epochs**: 5 (custom model), 1 (transformers, easily extensible)
- **Max Audio Length**: 16000 samples (1 second at 16kHz)

## Results

The notebook evaluates models using:
- **Accuracy**: Overall correctness
- **Confusion Matrix**: True/False positives and negatives
- **Precision, Recall, F1-Score**: Per-class performance metrics
- **Classification Report**: Detailed breakdown of metrics

  ![Image2](https://github.com/arshad-perampalli/Deepfake_Audio_Detection/blob/main/output2.png?raw=true)

  ---

  ![Image3](https://github.com/arshad-perampalli/Deepfake_Audio_Detection/blob/main/Screenshot%202025-06-28%20152344.png?raw=true)




## References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch Audio](https://pytorch.org/audio/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)


## Author

Arshad Perampalli
