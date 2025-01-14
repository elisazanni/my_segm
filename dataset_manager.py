from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import semantic_masks
from pathlib import Path
import os
import torch
import albumentations as albu
import yaml
from config import config



def to_tensor(x, **kwargs):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=2)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn, dim_images=512):
    _transform = [
        albu.Resize(dim_images, dim_images),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_augmentations():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(p=0.2),
        albu.Resize(height=512, width=512),
        albu.RandomRotate90(p=0.5)
    ])


class Histology_Dataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, augmentations=None, preprocessing=None):
        imgs_dir = Path(imgs_dir).resolve()
        masks_dir = Path(masks_dir).resolve()

        self.images = sorted(imgs_dir.glob('*_0000*.png'))
        self.masks = sorted(masks_dir.glob('*_mask*.png'))

        self.image_paths = []
        self.mask_paths = []

        self.preprocessing = preprocessing
        self.augmentations = augmentations

        mask_dict = {os.path.basename(mask_path).split('_mask')[0]+os.path.basename(mask_path).split('_mask')[1][:-4]: mask_path for mask_path in self.masks}

        for img_path in self.images:
            img_name = img_path.name
            base_name = img_name.split('_0000')[0] + img_name.split('_0000')[1][:-4] # Parte comune tra immagine e maschera
            #base_name_with_mask_word = base_name + '_mask' + img_name.split('_0000')[1][:-4]

            # Cerca la maschera con il base_name corrispondente, indipendentemente dal suffisso
            matching_mask = mask_dict.get(base_name)
            if matching_mask:
                self.image_paths.append(img_path)
                self.mask_paths.append(matching_mask)
            else:
                print(f"Warning: Mask for image '{img_name}' not found.")

    def __len__(self):
        #print("la lennnnnnnnnnnnnn:", len(self.image_paths))
        return len(self.image_paths)


    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image file not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), 0)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found at {mask_path}")

        if self.augmentations:
            sample = self.augmentations(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Remove channel dimension from the mask if necessary
        mask = mask.squeeze()  # Ensures no extra dimensions are left

        # Convert the mask to a LongTensor
        mask = torch.tensor(mask, dtype=torch.long)

        # Uncomment the following for one-hot encoding if needed for your model
        mask = torch.nn.functional.one_hot(mask, num_classes=len(config['classes'])).permute(2, 0, 1).float()

        return image, mask


    def show_image_mask_pair(self, index, linux_flag=False):
        if index < 0 or index >= len(self.image_paths):
            print(f"Index {index} is out of range.")
            return

        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        print(f"Image path: {img_path}")
        print(f"Mask path: {mask_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: Unable to read image file {img_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("image shape: ", img.shape)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Unable to read mask file {mask_path}")
            return
        print("mask shape: ", mask.shape)

        mask_color = semantic_masks.labels2colors(mask)

        img = img.astype(np.uint8)
        mask_color = mask_color.astype(np.uint8)

        if linux_flag:
            # Ensure the metrics directory exists
            os.makedirs('metrics', exist_ok=True)

            # Save the image
            img_fig, img_ax = plt.subplots()
            img_ax.imshow(img)
            img_ax.axis('off')
            img_fig.savefig(f'metrics/image_{index}.png', bbox_inches='tight')
            plt.close(img_fig)

            # Save the mask
            mask_fig, mask_ax = plt.subplots()
            mask_ax.imshow(mask_color)
            mask_ax.axis('off')
            mask_fig.savefig(f'metrics/mask_{index}.png', bbox_inches='tight')
            plt.close(mask_fig)

            print(f"Saved image and mask for index {index} in 'metrics' folder.")
        else:
            # Display the image and mask
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img)
            axes[0].set_title('Image')
            axes[0].axis('off')

            axes[1].imshow(mask_color)
            axes[1].set_title('Mask')
            axes[1].axis('off')

            plt.show()

