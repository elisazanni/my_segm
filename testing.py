import os
import cv2
import segmentation_models_pytorch as smp
import torch
from alive_progress import alive_bar
from src.utils.semantic_masks import class2color
import time
import numpy as np
from src.utils.albumentation import get_preprocessing
from config import config

IMAGE_HEIGHT = config['patch_size']
IMAGE_WIDTH = config['patch_size']

# If you use the binary model
# classes = ['background', 'instrument']
# If you use the multiclass model
# (see the train_model.py for the entire lists of classes)
# classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
#           'Suction', 'Large Needle Driver', 'Echography']
classes = config['classes']

weights_files = glob.glob(os.path.join(config['results_dir'], '*.pth'))

# Verifica se esiste almeno un file .pth
if weights_files:
    weights_path = weights_files[0]
else:
    raise FileNotFoundError("Nessun file con estensione .pth trovato nella directory dei risultati")


# Choose the encoder and the segmentation model
ENCODER = config['encoder']  # encoder
ENCODER_WEIGHTS = config['encoder_weights']  # pretrained weights
ACTIVATION = None
CLASSES = classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = config['model_name']  # segmentation model

# create segmentation architecture
if MODEL_NAME == 'FPN':
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(classes),
        activation=ACTIVATION,
        aux_params={'classes': len(classes), 'dropout': 0.38}
    )
elif MODEL_NAME == 'DeepLabV3+':
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        encoder_output_stride=16,
        classes=len(classes),
        activation=ACTIVATION,
        aux_params={'classes': len(classes), 'dropout': 0.38}
    )
elif MODEL_NAME == 'Unet++':
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        decoder_channels=[256, 128, 64, 32, 16],
        classes=len(classes),
        activation=ACTIVATION,
        aux_params={'classes': len(classes), 'dropout': 0.38}
    )
else:
    model = None

model.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda')))
model.to(DEVICE).eval()

preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)
preprocessing = get_preprocessing(preprocessing_fn)



