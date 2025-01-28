import numpy as np


colors_to_labels = {
    (0, 0, 0): 0,  # Background
    (0, 0, 200): 1,  # Tumor
    (50, 50, 50): 2,  # Necrosis
    (255, 255, 0): 3,  # Tissue
    (0, 128, 0): 4,  # Cirrhotic tissue
    (204, 102, 77): 5  # Exogenous material
}


labels_to_colors = {
    0: (0, 0, 0),
    1: (0, 0, 200),
    2: (50, 50, 50),
    3: (255, 255, 0),
    4: (0, 128, 0),
    5: (204, 102, 77)
}

id2label = {0: 'background', 1: 'Tumor', 2: 'Necrosis', 3:'Tissue', 4:'Cirrhotic tissue', 5:'Exogenous material'}
label2id = {v: k for k, v in id2label.items()}


def labels2colors(mask):
    """
    Function which generates a colored mask based on the input class value mask.
    :param mask: A mask where each class has its own integer value or a one-hot encoded mask.
    :return: A mask where each class has its own color.
    """
    # Se la maschera � 4D (batch, classi, H, W), rimuovi la dimensione batch e usa argmax
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0].argmax(axis=0)  # (6, 512, 512) -> (512, 512)

    # Se la maschera � 3D (classi, H, W), usa argmax
    elif mask.ndim == 3 and mask.shape[0] > 1:
        mask = mask.argmax(axis=0)  # (6, 512, 512) -> (512, 512)

    # Verifica che la maschera sia ora 2D (H, W)
    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}. Expected 2D mask.")

    # Inizializza un'immagine vuota per i colori
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3], dtype=np.uint8)

    # Applica i colori in base ai valori delle classi
    for i in labels_to_colors:
        color_mask[np.where(mask == i)] = labels_to_colors[i]

    return color_mask
