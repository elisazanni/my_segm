import os
import segmentation_models_pytorch as smp
import torch
import dataset_manager
import glob
from PIL import Image
import sys
from config import config

IMAGE_HEIGHT = config['patch_size']
IMAGE_WIDTH = config['patch_size']

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
DEVICE = config['device']
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

def predict_and_save(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE)))
    model.to(DEVICE).eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(config['encoder'], config['encoder_weights'])
    preprocessing = dataset_manager.get_preprocessing(preprocessing_fn)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('_0000.png'):
            # Carica l'immagine
            img_path = os.path.join(input_dir, file_name)
            img = Image.open(img_path)
            img_tensor = preprocessing(img).unsqueeze(0).to(DEVICE)

            # Predici
            with torch.no_grad():
                output = model(img_tensor)
                output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Salva l'immagine predetta
            pred_img = Image.fromarray(output.astype('uint8'))
            pred_file_name = file_name.replace('_0000.png', '_pred.png')
            pred_img.save(os.path.join(output_dir, pred_file_name))




input_directory = sys.argv[2]
output_directory = sys.argv[3]
predict_and_save(input_directory, output_directory)