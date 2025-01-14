import os
from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pandas as pd
import dataset_manager
from epoch import TrainEpoch, ValidEpoch
import matplotlib.pyplot as plt
import yaml
import sys
from dataset_manager import Histology_Dataset
from utils import saveResults, create_dataframe, create_testdataframe
from config import config


def choose_model():
    classes = config['classes']
    if config['model_name'] == 'FPN':
        model = smp.FPN(
            encoder_name=config['encoder'],
            encoder_weights=config['encoder_weights'],
            classes=len(classes),
            activation=None,
            aux_params={'classes': len(classes), 'dropout': 0.386}
        )
    elif config['model_name'] == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=config['encoder'],
            encoder_weights=config['encoder_weights'],
            encoder_output_stride=16,
            classes=len(classes),
            activation=None,
            aux_params={'classes': len(classes), 'dropout': 0.386}
        )
    elif config['model_name'] == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=config['encoder'],
            encoder_weights=config['encoder_weights'],
            decoder_channels=[256, 128, 64, 32, 16],
            classes=len(classes),
            activation=None,
            aux_params={'classes': len(classes), 'dropout': 0.43}
        )
    else:
        model = None

    return model


# LOSSES (2 losses if you want to experience with a combination of losses, otherwise just pass None)
def choose_loss():
    if config['loss'] == 'focal':
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
        loss2 = None
    elif config['loss'] == 'dice':
        loss = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        loss2 = None
    elif config['loss'] == "focaldice":
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
        loss2 = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
    else:
        loss = None
        loss2 = None

    return loss, loss2


if __name__ == '__main__':
    torch.cuda.empty_cache()
    model = choose_model()


    results_dir = Path(config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = results_dir.joinpath("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)


    model.to(config['device'])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(config['encoder'], config['encoder_weights'])
    model.to(config['device'])

    # Initialize datasets
    train_dataset = Histology_Dataset(
        config['train'], config['train_masks'],
        augmentations=dataset_manager.get_augmentations(),
        preprocessing=dataset_manager.get_preprocessing(preprocessing_fn),
    )
    print(f"Train dataset size: {len(train_dataset)}")

    valid_dataset = Histology_Dataset(
        config['val'], config['val_masks'],
        augmentations=dataset_manager.get_augmentations(),
        preprocessing=dataset_manager.get_preprocessing(preprocessing_fn),
    )
    print(f"Validation dataset size: {len(valid_dataset)}")

    test_dataset = Histology_Dataset(
        config['test'], config['test_masks'],
        augmentations=dataset_manager.get_augmentations(),
        preprocessing=dataset_manager.get_preprocessing(preprocessing_fn),
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # Dataset loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    loss, loss2 = choose_loss()

    metrics = []

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=config['learning_rate'])])

    # Scheduler for learning rate reduction
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-4, factor=0.7)

    train_epoch = TrainEpoch(
        model, loss=loss, loss2=loss2, metrics=metrics, optimizer=optimizer,
        classes=config['classes'], device=config['device'], verbose=True
    )

    valid_epoch = ValidEpoch(
        model, loss=loss, loss2=loss2, metrics=metrics, device=config['device'],
        verbose=True, classes=config['classes']
    )

    results_df = pd.DataFrame()  # DataFrame to store training stats
    es_counter = 0  # Early-Stopping counter
    optimal_val_loss = 1000  # Best validation loss
    max_score = 0  # Max mean IoU

    # Caricamento del checkpoint
    checkpoint_path = results_dir / f"{config['encoder']}-{config['model_name']}_checkpoint.pth"
    if checkpoint_path.exists():
        print(f"Caricamento del checkpoint da {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        es_counter = checkpoint['es_counter']
        optimal_val_loss = checkpoint['optimal_val_loss']
        max_score = checkpoint['max_score']
    else:
        print("Nessun checkpoint trovato. Inizio del training da zero.")
        start_epoch = 1

    for i in range(start_epoch, config['epochs'] + 1):
        print(f'\nEpoch: {i}')

        # Training e validazione
        train_logs = train_epoch.run(train_loader)
        valid_logs, valid_iou, valid_dice = valid_epoch.run(valid_loader)

        # Salvataggio dei risultati
        save = False
        if (i % 1) == 0:
            save = True

        IoU, inference_time, FBetaScore, DiceScore = saveResults(valid_loader, model, len(config['classes']),
                                                                 results_dir, save_img=save)

        curr_res = create_dataframe(model, i, IoU, inference_time, FBetaScore, config['classes'], train_logs,
                                    valid_logs, DiceScore)

        results_df = pd.concat([results_df, curr_res])
        results_df.to_excel(results_dir / f"{config['encoder']}-{config['model_name']}-{config['epochs']}ce.xlsx")

        # Early stopping e salvataggio del modello migliore
        if config['save_criterion'] == "validation_loss":
            if valid_logs['loss'] < optimal_val_loss:
                optimal_val_loss = valid_logs['loss']
                es_counter = 0
                torch.save(model.state_dict(),
                           results_dir / f"{config['encoder']}-{config['model_name']}-{config['epochs']}ce.pth")
                print('Model saved!')
            else:
                es_counter += 1
                print(f"EarlyStopping-counter = {es_counter}")

        elif config['save_criterion'] == "mean_IoU":
            curr_mean_IoU = np.mean(IoU)
            print('Current mean IoU: ', curr_mean_IoU)
            if curr_mean_IoU > max_score:
                max_score = curr_mean_IoU
                es_counter = 0
                torch.save(model.state_dict(),
                           results_dir / f"{config['encoder']}-{config['model_name']}-{config['epochs']}ce.pth")
                print('Model saved!')
            else:
                es_counter += 1
                print(f"EarlyStopping-counter = {es_counter}")

        # **Salvataggio dello stato completo**
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'es_counter': es_counter,
            'optimal_val_loss': optimal_val_loss,
            'max_score': max_score
        }, metrics_dir / f"{config['encoder']}-{config['model_name']}_checkpoint.pth")
        print(f"Checkpoint salvato per l'epoca {i}")

        # Early stopping
        if es_counter >= config['patience']:
            print(f"EARLY STOPPING... TRAINING IS STOPPED")
            break

        # Print IoU di ogni classe
        for c in range(len(config["classes"])):
            print(config['classes'][c], IoU[c])

        scheduler.step(valid_logs['loss'])  # Scheduler step basato sulla validation loss
        print('Ready for the next epoch')

        # Cleanup
        del train_logs, valid_logs, IoU

    # Test phase
    test_logs, test_iou, test_dice = valid_epoch.run(test_loader)
    IoU, inference_time, FBetaScore, DiceScore = saveResults(test_loader, model, len(config['classes']), metrics_dir, save_img=True)
    curr_res = create_testdataframe(model, config['epochs'], IoU, inference_time, FBetaScore, config['classes'], test_logs, DiceScore)
    results_df = pd.concat([results_df, curr_res])

    # Save loss and IoU plots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.plot(results_df['Valid Loss'], label='Valid Loss')
    ax0.plot(results_df['Train Loss'], label='Train Loss')
    ax0.legend(loc='best')
    ax1.plot(results_df['Mean IoU'])
    ax1.set_title('mIoU')
    plt.tight_layout()
    plt.savefig(metrics_dir / f"plots-{config['epochs']}.png")

    # Save the final training report
    results_df.to_excel(metrics_dir / f"{config['encoder']}-{config['model_name']}-{config['epochs']}ce.xlsx")