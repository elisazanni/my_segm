import cv2
import os
from alive_progress import alive_bar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from config import config


def main(source_folder):
    masks = [os.path.join(source_folder, file) for file in os.listdir(source_folder) if
             file.endswith(('.png', '.jpg', '.jpeg'))]
    masks.sort()

    num_classes = len(config['classes'])
    count_classes = [0] * num_classes  #pixel count per class
    occurrences_classes = [0] * num_classes  #occurrence count per class

    with alive_bar(len(masks)) as bar:
        for mask_path in masks:
            # Apri la maschera
            mask = cv2.imread(mask_path, 0)
            width, height = mask.shape
            flat = mask.reshape(width * height)

            classes = np.unique(flat)

            for c in classes:
                if c < num_classes:
                    count_classes[c] += np.sum(flat == c)
                    occurrences_classes[c] += 1
                else:
                    print(f"Valore di classe fuori range: {c} in {mask_path}")

            bar()

    df = pd.DataFrame({
        'Occurrences': occurrences_classes,
        'Pixel Count': count_classes
    }, index=config['classes'])

    print(df)

    df.to_excel(source_folder + '/class_distribution_final.xlsx')

    plt.figure(figsize=(10, 6))
    plt.bar(config['classes'], count_classes, color='skyblue', label='Pixel Count')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Pixel Distribution per Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()

    plt.savefig(source_folder + '/class_distribution_pixel_plot.png')

    #plt.show()


if __name__ == '__main__':
    main(config['train_masks'])
    main(config['val_masks'])
    main(config['test_masks'])

