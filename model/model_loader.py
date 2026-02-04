import tensorflow as tf
import keras
import streamlit as st
from huggingface_hub import hf_hub_download
from utils.config import HF_REPO_ID,HF_MODEL_FILENAME
from utils.logger import setup_logger

logger=setup_logger("ModelLoader")

@st.cache_resource

def load_model():
    try:
        import keras

        logger.info("Downloading model from Hugging Face Hub")
        model_path=hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            repo_type="model"
        )

        from keras.initializers import Orthogonal
        import keras.layers

        def patch_keras_layers():
            """Patches Keras layers to ignore unsupported arguments during deserialization."""
            
            # Patch InputLayer
            original_input_from_config = keras.layers.InputLayer.from_config
            def patched_input_from_config(config):
                # Keras 3 expects 'shape', not 'batch_shape'
                if 'batch_shape' in config and 'shape' not in config:
                    # batch_shape is usually [None, d1, d2, ...]
                    # shape should be (d1, d2, ...)
                    config['shape'] = config['batch_shape'][1:]
                
                # Remove arguments not supported in Keras 3
                for arg in ['optional', 'batch_shape', 'batch_input_shape']:
                    if arg in config:
                        config.pop(arg)
                return original_input_from_config(config)
            
            keras.layers.InputLayer.from_config = patched_input_from_config
            
            # Patch Dense
            original_dense_from_config = keras.layers.Dense.from_config
            def patched_dense_from_config(config):
                # Remove arguments not supported in Keras 3
                if 'quantization_config' in config:
                    config.pop('quantization_config')
                return original_dense_from_config(config)
            
            keras.layers.Dense.from_config = patched_dense_from_config

        # Apply patches
        patch_keras_layers()

        logger.info("Loading Tensorflow model")
        model=keras.models.load_model(model_path, custom_objects={'Orthogonal': Orthogonal})
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=['accuracy']
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc(file=sys.stderr)
        logger.exception("Failed to load model")
        st.error(f"Failed to load the model: {e}")
        # raise e # Don't raise, let app continue? No, app depends on model.
        raise e






