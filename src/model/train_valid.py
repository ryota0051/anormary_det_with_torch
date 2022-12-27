
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.functional import softmax

from src.labels import DATA_LABEL_DICT
from src.model.pytorchtools import EarlyStopping


class Trainer:
    def __init__(
            self,
            epochs: int,
            optimizer,
            loss_fn,
            model,
            positive_label=DATA_LABEL_DICT['good'],
            early_stopping: EarlyStopping = None):
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.positive_label = positive_label
        self.early_stopping = early_stopping

    def train(self, train_dl, valid_dl):
        history = {
            'loss': [],
            'auc': [],
            'val_loss': [],
            'val_auc': []
        }
        for epoch in range(self.epochs):
            loss, auc_score = self.train_one_step(train_dl)
            val_loss, val_auc_score = self.valid_one_step(valid_dl)
            if self.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    break
            history['loss'].append(loss)
            history['auc'].append(auc_score)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc_score)
            print(
                f'Epoch {epoch + 1} / {self.epochs} loss: {loss:.4f}, val_loss: {val_loss:.4f}, auc: {auc_score:.4f}, val_auc: {val_auc_score:.4f}'
            )
        return history

    def train_one_step(self, train_dl):
        self.model.train()
        result_loss = 0
        y_test, y_pred = [], []
        for x_batch, y_batch in train_dl:
            pred = self.model(x_batch)
            loss = self.loss_fn(pred, y_batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            result_loss += loss.item() * len(y_batch)
            pos_prob = softmax(pred, dim=-1)[:, self.positive_label].detach().cpu().numpy()
            y_test.append(y_batch.cpu().numpy().argmax(axis=1))
            y_pred.append(pos_prob)
        result_loss /= len(train_dl.dataset)
        result_auc = roc_auc_score(
            np.concatenate(y_test),
            np.concatenate(y_pred)
        )
        return result_loss, result_auc

    def valid_one_step(self, valid_dl):
        self.model.eval()
        result_loss = 0
        y_test, y_pred = [], []
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                result_loss += loss.item() * len(y_batch)
                pos_prob = softmax(pred, dim=-1)[:, self.positive_label].cpu().numpy()
                y_test.append(y_batch.cpu().numpy().argmax(axis=1))
                y_pred.append(pos_prob)
        result_loss /= len(valid_dl.dataset)
        result_auc = roc_auc_score(
            np.concatenate(y_test),
            np.concatenate(y_pred)
        )
        return result_loss, result_auc
