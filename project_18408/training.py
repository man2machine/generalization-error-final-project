# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:21 2021

@author: Shahir
"""

import os
import time
import datetime
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

def get_timestamp():
    return datetime.datetime.now().strftime("%m-%d-%Y %I-%M%p")

class ModelTracker:
    def __init__(self, root_dir): 
        experiment_dir = "Experiment {}".format(get_timestamp())
        self.save_dir = os.path.join(root_dir, experiment_dir)
        self.best_model_metric = float('-inf')
        self.record_per_epoch = {}
        os.makedirs(self.save_dir, exist_ok=True)
    
    def update_info_history(self,
                            epoch,
                            info):
        self.record_per_epoch[epoch] = info
        fname = "Experiment Epoch Info History.pckl"
        with open(os.path.join(self.save_dir, fname), 'wb') as f:
            pickle.dump(self.record_per_epoch, f)
    
    def update_model_weights(self,
                             epoch,
                             model_state_dict,
                             metric=None,
                             save_best=True,
                             save_current=True):
        update_best = metric is None or metric > self.best_model_metric
        if update_best and metric is not None:
            self.best_model_metric = metric
        
        if save_best and update_best:
            torch.save(model_state_dict, os.path.join(self.save_dir,
                "Weights Best.pckl"))
        if save_current:
                torch.save(model_state_dict, os.path.join(self.save_dir,
                    "Weights Epoch {} {}.pckl".format(epoch, get_timestamp())))

def make_optimizer(model, lr=0.001, weight_decay=0.0,
                   clip_grad_norm=False, verbose=False,
                   adam=False,
                   sgd=True):
    # Get all the parameters
    params_to_update = model.parameters()
    
    if verbose:
        print("Params to learn:")
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    
    if adam:
        optimizer = optim.Adam(params_to_update, lr=lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=True)
    elif sgd:
        optimizer = optim.SGD(params_to_update, lr=lr)
    else:
        raise ValueError()
    if clip_grad_norm:
        nn.utils.clip_grad_norm_(params_to_update, 3.0)
    
    return optimizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_optimizer_lr(optimizer,
                     lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_scheduler(optimizer, epoch_steps, gamma):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epoch_steps, gamma=gamma)
    return scheduler

def get_loss():
    criterion = nn.CrossEntropyLoss()
    return criterion

def train_model(
    device,
    model,
    dataloaders,
    criterion,
    optimizer,
    save_dir,
    lr_scheduler=None,
    save_model=False,
    save_best=False,
    save_all=False,
    save_log=False,
    num_epochs=1):
    
    start_time = time.time()
    
    tracker = ModelTracker(save_dir)
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        
        train_loss_info = {}

        print("Training")
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_count = 0
        
        train_loss_record = []
        pbar = tqdm(dataloaders['train'])
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, dim=1)

                # loss parts are for debugging purposes (if there are multiple components to a loss function)
                loss_parts = loss
                try:
                    iter(loss_parts)
                except TypeError:
                    loss_parts = [loss_parts]
                loss = sum(loss_parts)
                train_loss_record.append([n.detach().item() for n in loss_parts])
                
                loss.backward()
                optimizer.step()

                correct = torch.sum(preds == labels).item()
            
            running_loss += loss.detach().item() * inputs.size(0)
            running_correct += correct
            running_count += inputs.size(0)
            training_loss = running_loss / running_count
            training_acc = running_correct / running_count
            
            loss_fmt = "{:.4f}"
            desc = "Avg. Loss: {}, Total Loss: {}, Loss Parts: [{}]"
            desc = desc.format(loss_fmt.format(training_loss),
                                loss_fmt.format(sum(loss_parts)),
                                ", ".join(loss_fmt.format(n.item()) for n in loss_parts))
            pbar.set_description(desc)
            
            del loss, loss_parts
        pbar.close()

        print("Training Loss: {:.4f}".format(training_loss))
        print("Training Accuracy: {:.4f}".format(training_acc))
        train_loss_info['loss'] = train_loss_record
            
        print("Validation")
        model.eval()
        pbar = tqdm(dataloaders['val'])
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for inputs, labels in pbar:
            running_count += inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            correct = torch.sum(preds == labels).item()

            running_loss += F.cross_entropy(outputs, labels, reduction='mean').item()
            running_correct += correct
            val_accuracy = running_correct / running_count
            val_loss = running_loss * inputs.size(0) / running_count
        
        print("Validation loss {:.4f}".format(val_loss))
        print("Validation accuracy {:.4f}".format(val_accuracy))
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if save_model:
            model_weights = model.state_dict()
            tracker.update_model_weights(epoch,
                                         model_weights,
                                         metric=val_accuracy,
                                         save_best=save_best,
                                         save_current=save_all)
        
        if save_log:
            info = {'train_loss_history': train_loss_info}
            tracker.update_info_history(epoch, info)
        
        print()
        
        if lr_scheduler:
            lr_scheduler.step()

    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    
    return tracker

def save_training_session(model,
                          optimizer,
                          save_dir):
    sub_dir = "Session {}".format(get_timestamp())
    save_dir = os.path.join(save_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "Model State.pckl"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "Optimizer State.pckl"))
    
    print("Saved session to", save_dir)

def load_training_session(model,
                          optimizer,
                          session_dir,
                          update_models=True,
                          map_location=None):
    if update_models:
        model.load_state_dict(torch.load(os.path.join(session_dir, "Model State.pckl"), map_location=map_location))
        optimizer.load_state_dict(torch.load(os.path.join(session_dir, "Optimizer State.pckl"), map_location=map_location))
    
    print("Loaded session from", session_dir)

    out_data = {'model': model,
                'optimizer': optimizer}
    
    return out_data
