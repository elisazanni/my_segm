"""
Script which performs an Optuna study to choose the best hyperparameters of the model specified in the
input parameters. Usually here the hyperparameters to evaluate are: Dropout, Activation Function, Loss Function,
Optimizer, Learning Rate
"""

import sys
import optuna
import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from dataset_manager import Histology_Dataset
import dataset_manager
from torch.utils.data import DataLoader
from epoch import TrainEpoch, ValidEpoch
import yaml


optFile = sys.argv[1]


with open(optFile, "r") as f:
    opt = yaml.safe_load(f)

inFile = opt['config_file']
with open(inFile, "r") as f:
    config = yaml.safe_load(f)


BATCH_SIZE = int(config['batch_size'])
NUM_EPOCHS = int(opt['epochs'])
LOSS = config['loss']
SAVE_CRITERION = config['save_criterion']
ACTIVATION = None
#PROFILE = config['profile']
LEARNING_RATE = 1e-3
PATIENCE = config['patience']
classes = config['classes']
ENCODER = config['encoder']
ENCODER_WEIGHTS = config['encoder_weights']
CLASSES = classes
DEVICE = config['device']
MODEL_NAME = config['model_name']


def define_model(trial):

    # We optimize the dropout, the depth of the encoder and the activation function
    p = trial.suggest_float("dropout", 0.2, 0.5)
    #enc_depth = trial.suggest_int("encoder_depth", 3, 5)
    #activation = trial.suggest_categorical("activation", ["sigmoid", "softmax2d", "tanh"])

    # create segmentation model with pretrained encoder
    if MODEL_NAME == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            #encoder_depth=enc_depth,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes': len(classes), 'dropout': p}
        )
    elif MODEL_NAME == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            #encoder_depth=enc_depth,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes': len(classes), 'dropout': p}
        )
    elif MODEL_NAME == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            #encoder_depth=enc_depth,
            encoder_weights=ENCODER_WEIGHTS,
            decoder_channels=[256, 128, 64, 32, 16],
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes': len(classes), 'dropout': p}
        )
    else:
        model = None

    return model


def objective(trial):

    # Generate the model
    model = define_model(trial).to(DEVICE)

    # Generate optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    #Choose the loss
    loss = trial.suggest_categorical("loss", ["focal", "dice", "focaldice"])
    if loss == 'focal':
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
        loss2 = None
    elif loss == 'dice':
        loss = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        loss2 = None
    elif loss == 'focaldice':
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
        loss2= smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
    else:
        loss = None
        loss2 = None

    # define preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)

    #datset init
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

    # TRAIN AND VALIDATION LOADER
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # SCHEDULER for the reduction of the learning rate when the learning stagnates
    # namely when the valid loss doesn't decrease of a certain threshold for a fixed amount of epochs (patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=5e-4,
                                                           factor=0.2, verbose=True)

    metrics = [
        # smp.utils.metrics.IoU(threshold=0.5),
    ]

    # create epoch runners
    # an epoch is a loop which iterates over dataloader`s samples
    train_epoch = TrainEpoch(
        model,
        loss=loss,
        loss2=loss2,
        metrics=metrics,
        optimizer=optimizer,
        classes=classes,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        loss2=loss2,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
        classes=classes
    )

    for i in range(1, NUM_EPOCHS + 1):

        print('\nEpoch: {}'.format(i))

        # Run a train epoch
        train_logs = train_epoch.run(train_loader)
        # Run a validation epoch
        valid_logs, valid_iou, valid_dice = valid_epoch.run(valid_loader)

        IoU = np.mean(valid_iou)

        trial.report(IoU, i)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        scheduler.step(train_logs['loss'])

    return IoU



def main():
    study_name = opt['study_name'] # Nome dello studio
    db_file = f"{opt['study_name']}.db"  # Nome del file SQLite
    storage_name = f"sqlite:///{db_file}"  # Stringa di connessione SQLite

    # Controlla se il file SQLite esiste
    if os.path.exists(db_file):
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    else:
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")

    study.optimize(objective, n_trials=100, timeout=172800)
    
    # Salvare le visualizzazioni
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f'{opt["study_name"]}.png')

    # Stampa delle statistiche dello studio
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("     Number of finished trials: ", len(study.trials))
    print("     Number of pruned trials: ", len(pruned_trials))
    print("     Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("     Value:", trial.value)

    print("     Params: ")
    for key, value in trial.params.items():
        print("     {}: {}".format(key, value))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    out_dir = sys.argv[1]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main()