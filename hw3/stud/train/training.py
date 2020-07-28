import logging
import os

import torch
from tqdm.auto import tqdm


# from training.earlystopping import EarlyStopping


class Trainer:
    def __init__(self, model, loss_function, optimizer,
                 epochs, verbose, writer, _device, is_k_format):
        """
        Trainer object requires model, criterion (loss function), optimizer
        verbose level and number of epochs
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose
        self._epochs = epochs + 1
        self.writer = writer
        self.device = _device
        self.k_format = is_k_format

    def train(self, train_dataset, valid_dataset, save_to=None):
        """
        Trains the model, while keeping track of the best model trained so far
        and saves it if there is an improvement in the validation loss
        Applies Gradients clipping if their norm increased above the value specified
        in the function.
        Writes the model loss, val_Loss plots in tensorboard.
        Activates early stopping if needed with 5 epochs as patience to prevent overfitting

        Args:
            train_dataset (DataLoader)
            valid_dataset (DataLoader)
            save_to (str, optional): Save model to this path. Defaults to None.

        Returns:
            float: average training loss
        """
        train_loss, best_val_loss = 0.0, float(1e4)
        # es = EarlyStopping(patience=5)
        # lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)

        for epoch in range(1, self._epochs):
            epoch_loss = 0.0
            self.model.train()
            for _, sample in tqdm(enumerate(train_dataset),
                                  desc=f'Epoch {epoch}/{self._epochs - 1}',
                                  leave=False, total=len(train_dataset)):
                if self.k_format:
                    seq = sample["premises_hypotheses"].to(self.device)
                else:
                    premises_seq = sample["premises"].to(self.device)
                    hypotheses_seq = sample["hypotheses"].to(self.device)
                labels = sample["outputs"].to(self.device)
                labels_ = labels.view(-1)

                self.optimizer.zero_grad()
                predictions = self.model(premises_seq, hypotheses_seq) if not self.k_format else self.model(seq)
                sample_loss = self.loss_function(predictions, labels_)
                sample_loss.backward()

                # Gradient Clipping
                # clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()
                # To update tqdm bar with loss and val_loss
                # tqdm.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, refresh=True)
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            valid_loss = self.evaluate(valid_dataset)
            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', avg_epoch_loss)
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', valid_loss)

            # lr_scheduler.step(valid_loss)

            is_best = valid_loss <= best_val_loss
            if is_best:
                logging.info("Model Checkpoint saved")
                best_val_loss = valid_loss
                model_dir = os.path.join(os.getcwd(), 'model',
                                         f'{self.model.name}_ckpt_best')
                self.model.save_(model_dir)

            if self._verbose > 0:
                print(f'| Epoch: {epoch:02} | loss: {avg_epoch_loss:.4f} | val_loss: {valid_loss:.4f} |')

            # if es.step(valid_loss):
            #     print(f"Training Stopped early, epoch #: {epoch}")
            #     break

        avg_epoch_loss = train_loss / self._epochs
        if save_to is not None:
            self.model.save_(save_to)

        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        """
        Evaluation during training process to fetch val_loss & val_acc
        """
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, desc='Evaluating',
                               leave=False, total=len(valid_dataset)):
                if self.k_format:
                    seq = sample["premises_hypotheses"].to(self.device)
                else:
                    premises_seq = sample["premises"].to(self.device)
                    hypotheses_seq = sample["hypotheses"].to(self.device)
                labels = sample["outputs"].to(self.device)

                predictions = self.model(premises_seq, hypotheses_seq) if not self.k_format else self.model(seq)
                labels = labels.view(-1)

                sample_loss = self.loss_function(predictions, labels)
                valid_loss += sample_loss.tolist()
        return valid_loss / len(valid_dataset)
