import abc
import json
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import gamma
import matplotlib.pyplot as plt


class AnomalyDetector(abc.ABC):
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.threholds = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _get_model(self):
        return self.model
    
    @abc.abstractmethod
    def _create_model(self, args=None):
        exit(1)
    
    @abc.abstractclassmethod
    def get_input_shape(self):
        exit(1)
    
    def initialize(self):
        self.model = self._create_model().to(self.device)

    def load_or_train_model(self, dataset=None, val_dataset=None, is_train: bool=True):
        model_path_on_disk = self.args.model_path
        thresholds_path_on_disk = self.args.thresholds_path
        if is_train:
            self._train_model(dataset, val_dataset, model_path_on_disk)
            self._calc_and_store_thresholds(dataset, thresholds_path_on_disk)
        else:
            self._load_existing_model(model_path_on_disk)
            self._load_thresholds(thresholds_path_on_disk)
    
    def _load_existing_model(self, model_path_on_disk):
        self.model.load_state_dict(torch.load(model_path_on_disk))

    def _load_thresholds(self, thresholds_path_on_disk):
        pass
    
    def _train_model(self, dataset, val_dataset, model_path_on_disk):
            train_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8)
            val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, num_workers=8)
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
            criterion = nn.MSELoss()
            min_loss = float('inf')
            # self.model.load_state_dict(torch.load(model_path_on_disk))
            for epoch in range(1, 1 + self.args.epochs):
                self.model.train()
                pbar = tqdm(train_loader)
                for X in pbar:
                    batch_size = X.shape[0]
                    X = X.view(batch_size, -1).to(self.device)
                    if self.name == 'VAE':
                        mu_prime, mu, log_var = self.model(X)
                        loss = self.model.loss(X, mu_prime, mu, log_var)
                    else:
                        preds = self.model(X)
                        loss = criterion(X.view(batch_size, -1), preds)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))
                
                with torch.no_grad():
                    total_loss = 0
                    for X in val_loader:
                        batch_size = X.shape[0]
                        X = X.view(batch_size, -1).to(self.device)
                        if self.name == 'VAE':
                            mu_prime, mu, log_var = self.model(X)
                            loss = self.model.loss(X, mu_prime, mu, log_var)
                        else:
                            preds = self.model(X)
                            loss = criterion(X, preds)
                        total_loss += loss.item()
                    if min_loss > total_loss / len(val_loader):
                        min_loss = total_loss / len(val_loader)
                        torch.save(self.model.state_dict(), model_path_on_disk)
                        print("Epoch:", epoch, "Val loss:", total_loss / len(val_loader), "update!")
                    else:
                        print("Epoch:", epoch, "Val loss:", total_loss / len(val_loader))

    def single_test(self, ):
        pass
    
    def _calc_and_store_thresholds(self, dataset, thresholds_path_on_disk):
        data_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        print(self.args.batch_size)
        self.model.eval()
        pbar = tqdm(data_loader)
        criterion = nn.MSELoss(reduction="none")
        losses = np.array([])
        for X in pbar:
            batch_size = X.shape[0]
            X = X.view(batch_size, -1).to(self.device)
            preds = self.model(X)
            loss_batch = criterion(X.view(batch_size, -1), preds)
            loss_batch = torch.mean(loss_batch, dim=1)
            loss_batch = loss_batch.view(-1)
            loss_batch = loss_batch.cpu().detach().numpy()
            losses = np.hstack((losses, loss_batch))
        
        shape, loc, scale = gamma.fit(losses, floc=-0.1)
        thresholds = {}
        conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]
        for c in conf_intervals:
            thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)
        as_json = json.dumps(thresholds)
        print("Saving thresholds to %s" % thresholds_path_on_disk)
        if os.path.exists(thresholds_path_on_disk):
            os.remove(thresholds_path_on_disk)
        with open(thresholds_path_on_disk, 'a') as fp:
            fp.write(as_json)
