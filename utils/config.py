import os

#Tensorflow optimization flag
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

#Audio config
SAMPLE_RATE=22050
N_MFCC = 52

#Hugging Face model config
HF_REPO_ID = "ukRani03/Heart_murmur_model"
HF_MODEL_FILENAME= "final_heart_murmur_model.keras"

# Class Mapping
CLASSES = {
    0: "Artifact",
    1: "Murmur",
    2: "Normal"
}