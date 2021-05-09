# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:32 2021

@author: Shahir
"""

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch

def view_dataset_samples(dataset,
                         labeled=True,
                         tile_dims=(6, 3),
                         figsize=(8, 4)):
    fig = plt.figure(1, figsize=figsize)
    h, w = tile_dims
    for y in range(h):
        for x in range(w):
            n = y * w + x
            ax = fig.add_subplot(w, h, n + 1)
            if labeled:
                img, label = dataset[n]
            else:
                img = dataset[n]
            if img_transform:
                img = img_transform(img)
            ax.imshow(img)
            if labeled:
                ax.set_title(str(label))
            ax.axis('off')
    plt.show()

def get_model_predictions(dataloader, model, device, inputs_transform=None, labeled=True):
    pred_datas = []
    pbar = tqdm(dataloader)
    for data in pbar:
        if labeled:
            inputs = data[0]
        else:
            inputs = data
        if inputs_transform:
            inputs = inputs_transform(inputs)
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            pred_datas.append(preds.cpu().numpy())
    preds = np.concatenate(pred_datas)

    return preds

def get_model_accuracy(dataloader, model, device, inputs_transform=None, labeled=True):
    pbar = tqdm(dataloader)
    running_correct = 0
    running_count = 0
    for inputs, labels in pbar:
        if inputs_transform:
            inputs = inputs_transform(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            correct = torch.sum(preds == labels)

            running_correct += correct.item()
            running_count += inputs.size(0)
            accuracy = running_correct / running_count
    return accuracy

def show_model_predictions(dataloader,
                           model,
                           device,
                           inputs_transform=None,
                           labeled=True,
                           tile_dims=(3, 3),
                           figsize=(6, 6)):
    pbar = tqdm(dataloader)
    input_datas = []
    pred_datas = []
    output_datas = []
    label_datas = []

    num_imgs = np.prod(tile_dims)
    data_count = 0
    for data in pbar:
        if labeled:
            inputs, labels = data
        else:
            inputs = data
        if inputs_transform:
            inputs = inputs_transform(inputs)
        inputs = inputs.to(device)
        if labeled:
            labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

        outputs = torch.softmax(outputs, dim=1)
        input_datas.append(inputs.cpu().numpy())
        pred_datas.append(preds.cpu().numpy())
        output_datas.append(outputs.cpu().numpy())
        if labeled:
            label_datas.append(labels.cpu().numpy())

        data_count += inputs.size(0)
        if data_count >= np.prod(num_imgs):
            break
    
    inputs = np.concatenate(input_datas)
    preds = np.concatenate(pred_datas)
    outputs = np.concatenate(output_datas)
    if labeled:
        labels = np.concatenate(label_datas)

    fig = plt.figure(1, figsize=figsize)
    h, w = tile_dims
    for y in range(h):
        for x in range(w):
            n = y * w + x
            ax = fig.add_subplot(w, h, n + 1)
            img = inputs[n].transpose(1, 2, 0)
            ax.imshow(img)
            pred = preds[n]
            if labeled:
                label = labels[n]
                ax.set_title("True: {}\nPred: {}".format(label, pred))
            else:
                ax.set_title("Pred: {}".format(pred))
            ax.axis('off')
    plt.tight_layout()
    plt.show()

    data = {"inputs": inputs,
            "preds": preds,
            "outputs": outputs}
    if labeled:
        data["labels"] = labels
    
    return data
