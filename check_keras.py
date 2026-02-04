import keras
import tensorflow as tf
import sys
print(f"Keras Version: {keras.__version__}")
print(f"Keras File: {keras.__file__}")

try:
    from model.model_loader import load_model
    m = load_model()
    print("Model loaded successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
