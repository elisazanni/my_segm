import cv2
import numpy as np
import torch
import time
import semantic_masks
from torchsummaryX import summary
from torchmetrics.classification import MulticlassFBetaScore, BinaryFBetaScore
from torchmetrics import Dice
import pandas as pd
import os


def computeIoU(predictions, targets, num_classes):
    IoU = []
    for index in range(num_classes):
        # Reshape and concatenate each prediction and target individually
        reshaped_preds = [np.reshape(pred == index, (1, -1)) for pred in predictions]
        reshaped_targs = [np.reshape(targ == index, (1, -1)) for targ in targets]
        
        # Print shapes of individual reshaped predictions and targets
        #for i, (pred_shape, targ_shape) in enumerate(zip(reshaped_preds, reshaped_targs)):
        #    print(f"Prediction {i} shape: {pred_shape.shape}, Target {i} shape: {targ_shape.shape}")
        
        # Concatenate along axis=1 to align dimensions
        pred = np.concatenate(reshaped_preds, axis=1)
        targ = np.concatenate(reshaped_targs, axis=1)

        #print(f"Concatenated Prediction shape: {pred.shape}, Concatenated Target shape: {targ.shape}")

        # Compute IoU for the current class
        intersection = np.sum(np.bitwise_and(targ, pred))
        union = np.sum(np.bitwise_or(targ, pred))
        iou_class = (intersection / (union + 1e-10))
        IoU.append(iou_class)

    return IoU



def computeFBetaScore(predictions, targets, num_classes):
    if num_classes == 2:
        metric = BinaryFBetaScore(beta=1.0)
    else:
        metric = MulticlassFBetaScore(beta=1.0, num_classes=num_classes, average=None)

    return metric(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))


def computeDiceScore(predictions, targets):
    dice = Dice(average='micro')

    return dice(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))



# function to compute metrics and save the masks segmented by the network
def saveResults(X_test, model, num_classes, out_dir, save_img=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica il modello sulla device una volta sola
    model = model.to(device)

    ix = 0
    targets = []
    predictions = []
    comp_time = []

    for image, mask in X_test:
        # print(f"Original mask shape: {mask.shape}")
        targets.append(np.argmax(mask[0], axis=0).cpu().numpy())

        if hasattr(image, 'filename'):  # Check if the image has a filename attribute
            image_name = os.path.splitext(os.path.basename(image.filename))[0]
        else:
            image_name = f'image_{ix}'  # Fallback to a default name

        # Carica l'immagine sulla stessa device del modello
        image = image.to(device)

        start = time.time()
        res = np.argmax(model.predict(image)[0].cpu().squeeze(), axis=0)
        predictions.append(res.numpy())
        end = time.time()

        # Lista di tutti i tempi di inferenza
        comp_time.append(end - start)

        # Converte la maschera in una maschera colorata e la salva
        if save_img:
            save_dir = os.path.join(out_dir, 'TEST')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            # print(f"Prediction shape: {res.shape}")
            mask = semantic_masks.labels2colors(res)
            print("the saving path of the image is:", save_dir + f'/{image_name}_pred' + str(ix) + '.png')
            cv2.imwrite(save_dir + f'/{image_name}_pred' + str(ix) + '.png', mask[:, :, ::-1])
        ix = ix+1

    IoU = computeIoU(predictions, targets, num_classes)
    FBetaScore = np.array(computeFBetaScore(predictions, targets, num_classes))
    DiceScore = computeDiceScore(predictions, targets)

    return IoU, sum(comp_time) / len(comp_time), FBetaScore, DiceScore


# function to print summary and profiling of the network
def network_stats(model, device, batch_size):
    # Print summary of the model layers given the input dimension
    print(summary(model.to(device), torch.rand(batch_size, 3, 512, 512).to(device)))
    input = torch.rand(batch_size, 3, 512, 512).to(device)
    '''
    # warm-up
    model(input)
    # profile a network forward
    cuda = True if device == 'cuda' else 'False'
    with profiler.profile(with_stack=True, profile_memory=True, use_cuda=cuda) as prof:
        out = model(input)
    if cuda:
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total'))
    else:
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))
    '''


# function to create a dataframe on the valid epoch results
def create_dataframe(model, epoch, IoU, inference_time, FBetaScore, classes, train_logs, valid_logs, DiceScore):
    curr_res_dict = {}
    for j, v in enumerate(classes):
        dict_key = 'IoU_' + v.replace(' ', '')
        curr_res_dict[dict_key] = IoU[j]
    curr_res_dict['Mean IoU'] = np.mean(IoU)
    if len(classes) > 2:
        for j, v in enumerate(classes):
            dict_key = 'FBetaScore_' + v.replace(' ', '')
            curr_res_dict[dict_key] = FBetaScore[j]
        curr_res_dict['Mean FBetaScore'] = np.mean(FBetaScore)
    else:
        curr_res_dict['Binary FBetaScore'] = FBetaScore
    curr_res_dict['Dice Score'] = DiceScore
    curr_res_dict['Mean Inference Time'] = inference_time
    curr_res_dict['fps'] = 1 / inference_time
    curr_res_dict['No of Parameters'] = sum(p.numel() for p in model.parameters())
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    curr_res_dict['Memory(MB)'] = '{:.3f}MB'.format(size_all_mb)
    curr_res_dict['Train Loss'] = train_logs['loss']
    curr_res_dict['Valid Loss'] = valid_logs['loss']

    return pd.DataFrame(curr_res_dict, index=[epoch])


# function to create a dataframe on the final test epoch results
def create_testdataframe(model, num_epochs, IoU, inference_time, FBetaScore, classes, test_logs, DiceScore):
    curr_res_dict = {}
    for j, v in enumerate(classes):
        dict_key = 'IoU_' + v.replace(' ', '')
        curr_res_dict[dict_key] = IoU[j]
    curr_res_dict['Mean IoU'] = np.mean(IoU)
    if len(classes) > 2:
        for j, v in enumerate(classes):
            dict_key = 'FBetaScore_' + v.replace(' ', '')
            curr_res_dict[dict_key] = FBetaScore[j]
        curr_res_dict['Mean FBetaScore'] = np.mean(FBetaScore)
    else:
        curr_res_dict['Binary FBetaScore'] = FBetaScore
    curr_res_dict['Dice Score'] = DiceScore
    curr_res_dict['Mean Inference Time'] = inference_time
    curr_res_dict['fps'] = 1 / inference_time
    curr_res_dict['No of Parameters'] = sum(p.numel() for p in model.parameters())
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    curr_res_dict['Memory(MB)'] = '{:.3f}MB'.format(size_all_mb)
    curr_res_dict['Test Loss'] = test_logs['loss']

    return pd.DataFrame(curr_res_dict, index=[num_epochs + 1])