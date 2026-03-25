# Heart Murmur Detection using LSTM

A deep learning project to detect and classify heart murmurs from heartbeat audio recordings using a Long Short-Term Memory (LSTM) neural network. This repository includes a web application built with Streamlit for a user-friendly interface to process and analyze audio files in real-time.

## 🚀 Features

- **Audio Pattern Analysis**: Upload heartbeat sounds (`.wav` or `.mp3`) and view their waveforms.
- **Deep Learning Model**: Uses a trained LSTM model to classify sounds into 3 categories: **Artifact**, **Murmur**, and **Normal**.
- **Real-Time Classification**: Automatically extracts Mel-Frequency Cepstral Coefficients (MFCCs) and processes them for inference.
- **Hugging Face Integration**: Securely downloads and caches the pre-trained Keras model from the Hugging Face Hub automatically.

## 💻 Tech Stack

- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **Audio Processing**: Librosa, Soundfile
- **Web App UI**: Streamlit
- **Data Handling & Visualization**: NumPy, Pandas, Matplotlib
- **Model Hosting**: Hugging Face Hub

## 📂 Project Structure

- `app.py`: Main Streamlit application file.
- `audio/`: Audio loading and MFCC feature extraction scripts.
- `model/`: Scripts for downloading and loading the trained model from Hugging Face.
- `ui/`: Visualization utilities for plotting waveforms in the Streamlit app.
- `utils/`: Configuration (constants, class mappings) and logging setup.
- `Heart_murmur_project.ipynb`: Jupyter notebook used for data exploration, preprocessing, model creation, and training.
- `requirements.txt`: Required Python dependencies.
- `check_keras.py`: Utility script to verify Keras installation and model loading.

## 🛠️ Setup & Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository_url>
   cd Heart_Murmur_project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

To run the Streamlit web application locally:

```bash
streamlit run app.py
```

This will launch the app in your default web browser (typically at `http://localhost:8501`). From there, you can upload your heartbeat `.wav` or `.mp3` files and get instant classification results.

## 📊 Dataset and Resources

- **Dataset**: The model was trained on the [Heartbeat Sounds Dataset from Kaggle](https://www.kaggle.com/datasets/abdallahaboelkhair/heartbeat-sound).
- **Pre-trained Model**: The best-performing model is hosted on Hugging Face: [ukRani03/Heart_murmur_model](https://huggingface.co/ukRani03/Heart_murmur_model/blob/main/final_heart_murmur_model.keras).
