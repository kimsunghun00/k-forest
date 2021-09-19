import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
import tqdm
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import argparse

from model import Model


class Trainer:
    def __init__(self, net, train_loader, test_loader, criterion, optimizer, epochs=100, lr = 0.001,
                 l2_norm = None, device=None):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters(), lr=lr)
        self.epochs = epochs
        self.l2_norm = l2_norm

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.train_losses = []
        self.val_losses = []

        self.best_model_wts = copy.deepcopy(self.net.state_dict())
        self.best_mse = 100.

    def fit(self):
        lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.9)
        for epoch in range(self.epochs):
            running_loss = 0.
            self.net.train()

            for i, (X_batch, y_batch) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                X_batch = X_batch.to(self.device, dtype=torch.float)
                y_batch = y_batch.to(self.device, dtype=torch.float).reshape(-1, 1)
                y_pred_batch = self.net(X_batch)

                # regularization
                if self.l2_norm is not None:
                    lambda2 = self.l2_norm
                    fc_params = torch.cat([x.view(-1) for x in self.net.out.parameters()])
                    l2_regularization = lambda2 * torch.norm(fc_params, p=2)
                else:
                    l2_regularization = 0.

                loss = self.criterion(y_pred_batch, y_batch) + l2_regularization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            lr_scheduler.step()
            self.train_losses.append(running_loss / i)

            # 검증 데이터의 예측 정확도 및 Loss
            self.val_losses.append(self.eval_net(self.test_loader, self.device))

            # epoch 결과 표시
            print('epoch: {}/{}, train_loss: {:.8f}, val_loss: {:.8f}'.format(epoch + 1, self.epochs,
                                                                               self.train_losses[-1],
                                                                               self.val_losses[-1]))
        print('best mse : {:.8f}'.format(self.best_mse))

    def eval_net(self, data_loader, device):
        running_loss = 0.
        self.net.eval()
        for i, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device, dtype=torch.float).reshape(-1, 1)

            with torch.no_grad():
                y_pred_batch = self.net(X_batch)
                loss = self.criterion(y_pred_batch, y_batch)
            running_loss += loss.item()
        val_loss = running_loss / i

        if val_loss < self.best_mse:
            self.best_mse = val_loss
            self.best_model_wts = copy.deepcopy(self.net.state_dict())

        return val_loss

    def evaluation(self, data_loader, device):
        model = self.get_best_model()
        running_loss =0.
        for i, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device, dtype=torch.float).reshape(-1, 1)

            with torch.no_grad():
                y_pred_batch = model(X_batch)
                loss = self.criterion(y_pred_batch, y_batch)
            running_loss += loss.item()
        test_loss = running_loss / i
        print('mse :', test_loss)

    def history(self):
        return {'train_loss' : self.train_losses, 'val_loss' : self.val_losses}

    def get_best_model(self):
        self.net.load_state_dict(self.best_model_wts)
        return self.net