# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:37:48 2021

@author: Shahir
"""

import os
import json
import pickle
import hashlib
import shutil

import torch
from torch.utils.data import dataloader

from project_18408.utils import JSONDictSerializable
from project_18408.datasets import (ImageDatasetType, IMG_DATASET_TO_NUM_SAMPLES, IMG_DATASET_TO_NUM_CLASSES,
    get_img_dataset, get_sklearn_dataset, get_dataloaders,
    apply_corrupted_labels, apply_img_transforms)
from project_18408.training import (OptimizerType, LossType, make_optimizer, make_scheduler,
    save_training_session, load_training_session, get_loss, train_model, load_weights)
from project_18408.models.relu_toy_models import ReLUToyModel
from project_18408.evaluation import get_dataloader_stats

class ImageDatasetConfig(JSONDictSerializable):
    def __init__(self,
                 img_dataset_type,
                 num_train_samples=None,
                 num_test_samples=None,
                 new_input_size=None,
                 flatten=True,
                 augment=False,
                 corrupt_frac=0,
                 seed=0):
        self.img_dataset_type = img_dataset_type
        self.num_classes = IMG_DATASET_TO_NUM_CLASSES[img_dataset_type]
        self.num_train_samples = num_train_samples or IMG_DATASET_TO_NUM_SAMPLES[img_dataset_type][0]
        self.num_test_samples = num_test_samples or IMG_DATASET_TO_NUM_SAMPLES[img_dataset_type][1]
        self.new_input_size = new_input_size
        self.flatten = flatten
        self.augment = augment
        self.corrupt_frac = corrupt_frac
        self.seed = seed

    def to_dict(self):
        return {'img_dataset_type': self.img_dataset_type,
                'num_train_samples': self.num_train_samples,
                'num_test_samples': self.num_test_samples,
                'new_input_size': self.new_input_size,
                'flatten': self.flatten,
                'augment': self.augment,
                'corrupt_frac': self.corrupt_frac,
                'seed': self.seed}

    @classmethod
    def from_dict(cls, dct):
        return cls(img_dataset_type=dct['img_dataset_type'],
                   num_train_samples=dct['num_train_samples'],
                   num_test_samples=dct['num_test_samples'],
                   new_input_size=dct['new_input_size'],
                   flatten=dct['flatten'],
                   augment=dct['augment'],
                   corrupt_frac=dct['corrupt_frac'],
                   seed=dct['seed'])

    def generate_dataset(self, data_dir):
        datasets_orig = get_img_dataset(data_dir, self.img_dataset_type)
        datasets = apply_img_transforms(datasets_orig, self.img_dataset_type,
            new_input_size=self.new_input_size,
            augment=self.augment,
            flatten=self.flatten)
        if self.corrupt_frac > 0:
            datasets = apply_corrupted_labels(datasets, self.num_classes,
                corrupt_frac=self.corrupt_frac, seed=self.seed)
        
        return {'datasets': datasets,
                'datasets_img_orig': datasets_orig}

class SKLearnDatasetConfig(JSONDictSerializable):
    def __init__(self,
                 num_train_samples=60000,
                 num_test_samples=10000,
                 corrupt_frac=0,
                 num_features=2,
                 num_informative=0,
                 num_redundant=0,
                 num_repeated=0,
                 num_classes=2,
                 num_clusters_per_class=1,
                 seed=0):
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.corrupt_frac = corrupt_frac
        self.num_features = num_features
        self.num_informative = num_informative
        self.num_redundant = num_redundant
        self.num_repeated = num_repeated
        self.num_classes = num_classes
        self.num_clusters_per_class = num_clusters_per_class
        self.seed = seed

    def to_dict(self):
        return {'num_train_samples': self.num_train_samples,
                'num_test_samples': self.num_test_samples,
                'corrupt_frac': self.corrupt_frac,
                'num_features': self.num_features,
                'num_informative': self.num_informative,
                'num_redundant': self.num_redundant,
                'num_repeated': self.num_repeated,
                'num_classes': self.num_classes,
                'num_clusters_per_class': self.num_clusters_per_class,
                'seed': self.seed}

    @classmethod
    def from_dict(cls, dct):
        return cls(num_train_samples=dct['num_train_samples'],
                   num_test_samples=dct['num_test_samples'],
                   corrupt_frac=dct['corrupt_frac'],
                   num_features=dct['num_features'],
                   num_informative=dct['num_informative'],
                   num_redundant=dct['num_redundant'],
                   num_repeated=dct['num_repeated'],
                   num_classes=dct['num_classes'],
                   num_clusters_per_class=dct['num_clusters_per_class'],
                   seed=dct['seed'])

    def generate_dataset(self):
        datasets = get_sklearn_dataset(
            num_train_samples=self.num_train_samples,
            num_test_samples=self.num_test_samples,
            num_features=self.num_features,
            num_informative=self.num_informative,
            num_redundant=self.num_redundant,
            num_repeated=self.num_repeated,
            num_classes=self.num_classes,
            num_clusters_per_class=self.num_clusters_per_class,
            seed=self.seed)
        if self.corrupt_frac > 0:
            datasets = apply_corrupted_labels(datasets, self.num_classes,
                corrupt_frac=self.corrupt_frac, seed=self.seed)
        
        return {'datasets': datasets}

class DatasetType:
    IMG = "img"
    SKLEARN = "sklearn"

class DatasetConfig(JSONDictSerializable):
    def __init__(self,
                 dataset_type,
                 dataset_config=None):
        assert dataset_type in (DatasetType.IMG, DatasetType.SKLEARN)
        self.dataset_type = dataset_type
        self.config = dataset_config

    def to_dict(self):
        return {'dataset_type': self.dataset_type,
                'dataset_config': self.config.to_dict()}

    @classmethod
    def from_dict(cls, dct):
        dataset_type = dct['dataset_type']
        if dataset_type == DatasetType.IMG:
            dataset_config = ImageDatasetConfig.from_dict(dct['dataset_config'])
        elif dataset_type == DatasetType.SKLEARN:
            dataset_config = SKLearnDatasetConfig.from_dict(dct['dataset_config'])
        else:
            raise ValueError()
        return cls(dataset_type=dataset_type,
                   dataset_config=dataset_config)

    def generate_setup(self,
                       data_dir,
                       train_batch_size,
                       test_batch_size,
                       num_workers=4,
                       pin_memory=False):
        if self.dataset_type == DatasetType.IMG:
            setup = self.config.generate_dataset(data_dir)
        elif self.dataset_type == DatasetType.SKLEARN:
            setup = self.config.generate_dataset()
        else:
            raise ValueError()
        dataloaders = get_dataloaders(
            setup['datasets'],
            train_batch_size,
            test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory)
        setup['dataloaders'] = dataloaders
        return setup

class ReLUModelConfig(JSONDictSerializable):
    def __init__(self,
                 input_dim,
                 output_dim,
                 layer_dims,
                 bias=False,
                 seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dims = list(layer_dims)
        self.bias = bias
        self.seed = seed
    
    def to_dict(self):
        return {'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'layer_dims': self.layer_dims,
                'bias': self.bias,
                'seed': self.seed}

    @classmethod
    def from_dict(cls, dct):
        return cls(input_dim=dct['input_dim'],
                   output_dim=dct['output_dim'],
                   layer_dims=dct['layer_dims'],
                   bias=dct['bias'],
                   seed=dct['seed'])

    def generate_model(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        model = ReLUToyModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            layer_dims=self.layer_dims,
            bias=self.bias
        )

        return model

class ModelType:
    RELU_TOY = "relu_toy"

class ModelConfig(JSONDictSerializable):
    def __init__(self,
                 model_type,
                 model_config):
        assert model_type in (ModelType.RELU_TOY,)
        self.model_type = model_type
        self.config = model_config
    
    def to_dict(self):
        return {'model_type': self.model_type,
                'model_config': self.config.to_dict()}

    @classmethod
    def from_dict(cls, dct):
        model_type = dct['model_type']
        if model_type == ModelType.RELU_TOY:
            model_config = ReLUModelConfig.from_dict(dct['model_config'])
        else:
            raise ValueError()
        return cls(model_type=model_type,
                   model_config=model_config)

    def generate_model(self):
        if self.model_type == ModelType.RELU_TOY:
            return self.config.generate_model()
        else:
            raise ValueError()

class TrainingConfig(JSONDictSerializable):
    def __init__(self,
                 optimizer_type,
                 loss_type,
                 lr=0.01,
                 num_epochs=100,
                 clip_grad_norm=False,
                 weight_decay=0.0,
                 use_lr_schedule=False,
                 epoch_lr_decay_steps=None,
                 lr_decay_gamma=None,
                 early_stop=True,
                 early_stop_acc=0.80,
                 early_stop_patience=5):
        assert optimizer_type in (OptimizerType.ADAM, OptimizerType.SGD, OptimizerType.SGD_MOMENTUM)
        assert loss_type in (LossType.CROSS_ENTROPY)
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.lr = lr
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.weight_decay = weight_decay
        self.use_lr_schedule = use_lr_schedule
        self.epoch_lr_decay_steps = epoch_lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma
        self.early_stop = early_stop
        self.early_stop_acc = early_stop_acc
        self.early_stop_patience = early_stop_patience

        if use_lr_schedule:
            assert epoch_lr_decay_steps is not None
            assert lr_decay_gamma is not None
    
    def to_dict(self):
        return {'optimizer_type': self.optimizer_type,
                'loss_type': self.loss_type,
                'lr': self.lr,
                'num_epochs': self.num_epochs,
                'clip_grad_norm': self.clip_grad_norm,
                'weight_decay': self.weight_decay,
                'use_lr_schedule': self.use_lr_schedule,
                'epoch_lr_decay_steps': self.epoch_lr_decay_steps,
                'lr_decay_gamma': self.lr_decay_gamma,
                'early_stop': self.early_stop,
                'early_stop_acc': self.early_stop_acc,
                'early_stop_patience': self.early_stop_patience}

    @classmethod
    def from_dict(cls, dct):
        return cls(optimizer_type=dct['optimizer_type'],
                   loss_type=dct['loss_type'],
                   lr=dct['lr'],
                   num_epochs=dct['num_epochs'],
                   clip_grad_norm=dct['clip_grad_norm'],
                   weight_decay=dct['weight_decay'],
                   epoch_lr_decay_steps=dct['epoch_lr_decay_steps'],
                   lr_decay_gamma=dct['lr_decay_gamma'],
                   early_stop=dct['early_stop'],
                   early_stop_acc=dct['early_stop_acc'],
                   early_stop_patience=dct['early_stop_patience'])

    def generate_setup(self, model, device):
        optimizer = make_optimizer(model, lr=self.lr, weight_decay=self.weight_decay,
            clip_grad_norm=self.clip_grad_norm, optimzer_type=self.optimizer_type)
        scheduler = None
        if self.use_lr_schedule:
            scheduler = make_scheduler(optimizer, self.epoch_lr_decay_steps, self.lr_decay_gamma)
        criterion = get_loss(self.loss_type)
        criterion = criterion.to(device)
        
        return {'optimizer': optimizer,
                'criterion': criterion,
                'lr_scheduler': scheduler}

class ExperimentConfig(JSONDictSerializable):
    def __init__(self,
                 dataset_config,
                 model_config,
                 training_config,
                 trial_index=None):
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.training_config = training_config
        self.trial_index = trial_index
    
    def to_dict(self):
        return {'dataset_config': self.dataset_config.to_dict(),
                'model_config': self.model_config.to_dict(),
                'training_config': self.training_config.to_dict(),
                'trial_index': self.trial_index}
    
    @classmethod
    def from_dict(cls, dct):
        return cls(dataset_config=DatasetConfig.from_dict(dct['dataset_config']),
                   model_config=ModelConfig.from_dict(dct['model_config']),
                   training_config=TrainingConfig.from_dict(dct['training_config']),
                   trial_index=dct['trial_index'])

    def generate_setup(self,
                       data_dir,
                       device,
                       train_batch_size,
                       test_batch_size,
                       num_workers=4,
                       pin_memory=False):
        data_setup = self.dataset_config.generate_setup(
            data_dir,
            train_batch_size,
            test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        model = self.model_config.generate_model()
        model = model.to(device)
        training_setup = self.training_config.generate_setup(model, device)

        return {'data_setup': data_setup,
                'model': model,
                'training_setup': training_setup}

class ExperimentState(JSONDictSerializable):
    def __init__(self,
                 training_complete=False,
                 stats_complete=False):
        self.training_complete = training_complete
        self.stats_complete = stats_complete
    
    def to_dict(self):
        return {'training_complete': self.training_complete,
                'stats_complete': self.stats_complete}
    
    @classmethod
    def from_dict(cls, dct):
        return cls(training_complete=dct['training_complete'],
                   stats_complete=dct['stats_complete'])

class ExperimentManager:
    WEIGHTS_DIR = "weights"
    SESSION_DIR = "sessions"
    STATS_DIR = "stats"

    def __init__(self, data_dir, experiment_dir):
        self.data_dir = data_dir
        self.experiment_dir = experiment_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self._load_index()

    def _load_index(self):
        index_fname = os.path.join(self.experiment_dir, "index.json")
        if os.path.exists(index_fname):
            with open(index_fname, 'r') as f:
                index = json.load(f)
            self.index = index
        else:
            self.index = {}
            self._save_index()

    def _save_index(self):
        index_fname = os.path.join(self.experiment_dir, "index.json")
        with open(index_fname, 'w') as f:
            json.dump(self.index, f, indent=4)
    
    def _save_config(self, config, workspace_dir):
        config_fname = os.path.join(workspace_dir, "config.json")
        with open(config_fname, 'w') as f:
            json.dump(config.to_dict(), f, indent=4)
    
    def _load_state(self, workspace_dir):
        state_fname = os.path.join(workspace_dir, "state.json")
        with open(state_fname, 'r') as f:
            state = json.load(f)
        state = ExperimentState.from_dict(state)
        return state
    
    def _save_state(self, state, workspace_dir):
        state_fname = os.path.join(workspace_dir, "state.json")
        with open(state_fname, 'w') as f:
            json.dump(state.to_dict(), f, indent=4)
    
    def _save_stats(self, stats, workspace_dir):
        state_fname = os.path.join(workspace_dir, "stats.pckl")
        with open(state_fname, 'wb') as f:
            pickle.dump(stats, f)
    
    def _get_workspace_dir_from_hash(self, h):
        return os.path.join(self.experiment_dir, h)
    
    def get_workspace_dir(self, config):
        h = self.find_experiment(config)
        if not h:
            raise ValueError("Experiment not found")
        return self._get_workspace_dir_from_hash(h)

    def find_experiment(self, config):
        config_bytes = config.to_bytes()
        m = hashlib.blake2b(digest_size=8)
        m.update(config_bytes)
        h = m.hexdigest()
        if h in self.index:
            return h
        else:
            for h, c in self.index.items():
                if c == config.to_dict():
                    return h
            return False

    def add_experiment(self, config, exist_ok=False):
        config_bytes = config.to_bytes()
        experiment_exists = self.find_experiment(config)
        if experiment_exists and exist_ok:
            return experiment_exists
        elif experiment_exists and not exist_ok:
            raise ValueError()
        num_salt = 0
        while True:
            m = hashlib.blake2b(digest_size=8)
            m.update(config_bytes)
            if num_salt:
                m.update('a' * num_salt)
            h = m.hexdigest()
            if h in self.index:
                num_salt += 1
                continue
            break
        self.index[h] = config.to_dict()
        self._save_index()
        workspace_dir = self._get_workspace_dir_from_hash(h)
        os.makedirs(workspace_dir)
        state = ExperimentState()
        self._save_config(config, workspace_dir)
        self._save_state(state, workspace_dir)

        return h
    
    def remove_experiment(self, config):
        h = self.find_experiment(config)
        self.index.pop(h)
        workspace_dir = self._get_workspace_dir_from_hash(h)
        shutil.rmtree(workspace_dir)

    def load_experiment(self,
                        config,
                        device,
                        train_batch_size=128,
                        test_batch_size=128,
                        num_workers=4,
                        pin_memory=False,
                        load_from_session=True,
                        override_best=True):
        workspace_dir = self.get_workspace_dir(config)
        setup = config.generate_setup(
            self.data_dir,
            device,
            train_batch_size,
            test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        state = self._load_state(workspace_dir)
        if state.training_complete and load_from_session:
            model = setup['model']
            optimizer = setup['training_setup']['optimizer']
            sessions_dir = os.path.join(workspace_dir, self.SESSION_DIR)
            s = os.listdir(sessions_dir)
            assert len(s) == 1
            session_sub_dir = os.path.join(sessions_dir, s[0])
            out = load_training_session(
                model,
                optimizer,
                session_sub_dir)
            setup['model'] = out['model']
            setup['training_setup']['optimizer'] = out['optimizer']

            if override_best:
                weights_dir = os.path.join(workspace_dir, self.WEIGHTS_DIR)
                w = os.listdir(weights_dir)
                assert len(w) == 1
                weights_sub_dir = os.path.join(weights_dir, w[0])
                weights_fname = os.path.join(weights_sub_dir, "Weights Best.pckl")
                load_weights(setup['model'], weights_fname)
        
        return setup, state
    
    def run_training(self,
                     config,
                     device,
                     train_batch_size=128,
                     test_batch_size=128,
                     num_workers=4,
                     pin_memory=False,
                     completed_ok=False):
        setup, state = self.load_experiment(
            config,
            device,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            load_from_session=True,
            override_best=False
        )
        if state.training_complete:
            if completed_ok:
                return setup, state
            else:
                raise ValueError("Training already complete")

        workspace_dir = self.get_workspace_dir(config)
        weights_dir = os.path.join(workspace_dir, self.WEIGHTS_DIR)
        sessions_dir = os.path.join(workspace_dir, self.SESSION_DIR)

        datasets = setup['data_setup']['datasets']
        dataloaders = setup['data_setup']['dataloaders']
        model = setup['model']
        optimizer = setup['training_setup']['optimizer']
        lr_scheduler = setup['training_setup']['lr_scheduler']
        criterion = setup['training_setup']['criterion']
        num_epochs = config.training_config.num_epochs
        
        early_stop = config.training_config.early_stop
        early_stop_acc = config.training_config.early_stop_acc
        early_stop_patience = config.training_config.early_stop_patience

        model = model.to(device)
        criterion = criterion.to(device)

        tracker = train_model(
            device=device,
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            save_dir=weights_dir,
            lr_scheduler=lr_scheduler,
            save_model=True,
            save_best=True,
            num_epochs=num_epochs,
            early_stop=early_stop,
            early_stop_acc=early_stop_acc,
            early_stop_patience=early_stop_patience
        )
        # no need for save latest as we are saving the session anyways

        setup['model_tracker'] = tracker
        save_training_session(model, optimizer, sessions_dir)
        state.training_complete = True
        self._save_state(state, workspace_dir)

        return setup, state

    def run_stats(self, config, device, save_stats=False):
        setup, state = self.load_experiment(config, device)
        workspace_dir = self.get_workspace_dir(config)
        dataloaders = setup['data_setup']['dataloaders']
        model = setup['model']
        criterion = setup['training_setup']['criterion']
        train_stats = get_dataloader_stats(dataloaders['train'], model, criterion, device)
        test_stats = get_dataloader_stats(dataloaders['test'], model, criterion, device)

        stats = {'train': train_stats,
                 'test': test_stats}
        
        if save_stats:
            state.stats_complete = True
            self._save_stats(stats, workspace_dir)
            self._save_state(state, workspace_dir)
        
        return stats

        