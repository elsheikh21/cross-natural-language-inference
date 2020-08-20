import nlp
import time
import logging
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from utils import compute_epoch_time
from train.earlystopping import EarlyStopping
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
import pkbar


class Trainer:
    def __init__(self, model, loss_function, optimizer,
                 epochs, verbose, writer, _device, is_k_format):
        """
        Trainer object requires model, criterion (loss function), optimizer
        verbose level and number of epochs. Used either to train BaselineModel or the K_Model
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
        es = EarlyStopping(patience=5)
        lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        tik = time.time()
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
                clip_grad_norm_(self.model.parameters(), 5.)
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

            lr_scheduler.step(valid_loss)

            is_best = valid_loss <= best_val_loss
            if is_best:
                logging.info("Model Checkpoint saved")
                best_val_loss = valid_loss
                model_dir = os.path.join(os.getcwd(), 'model',
                                         f'{self.model.name}_ckpt_best')
                self.model.save_(model_dir)

            if self._verbose > 0:
                tok = time.time()
                tiktok = compute_epoch_time(tok - tik)
                print(
                    f'| Epoch: {epoch:02} | loss: {avg_epoch_loss:.4f} | val_loss: {valid_loss:.4f} | time: {tiktok} |')

            if es.step(valid_loss):
                print(f"Training Stopped early, epoch #: {epoch}")
                break

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


class BERT_Trainer:
    def __init__(self, model, loss_function, optimizer,
                 epochs, verbose, writer, _device):
        """
        Trainer object requires model, criterion (loss function), optimizer
        verbose level and number of epochs. Used to train the mBERT model
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose
        self._epochs = epochs
        self.writer = writer
        self.device = _device

    def train(self, train_dataset, valid_dataset, save_to=None):
        """
        Trains the model, while keeping track of the best model trained so far
        and saves it if there is an improvement in the validation loss
        Applies Gradients clipping if their norm increased above the 1.0
        Writes the model loss, val_Loss plots in tensorboard.
        Activates early stopping if needed with 5 epochs as patience to prevent overfitting

        It uses pkbar package to have a progress bar keras like,
        the progress bar shows the loss, val_loss during training as well as acc & val_acc
        In order to compute the accuracy, we make use from nlp metric library for XNLI tasks. the accuracy is computed
        from model's predictions during training and the original labels, the predictions are fetched as the argmax of
        the logits tensor.

        It initializes a linear scheduler with a warm up for the trainer's optimizer that helps with increases the
        learning rate linearly from 0 to the initial lr set in the optimizer

        Model takes 3 arguments (from the sequences padded as per batch) to do the forward pass:
        1. Sequence (inputs_ids)
        2. Attention masks
        3. Token types

        Args:
            train_dataset (DataLoader)
            valid_dataset (DataLoader)
            save_to (str, optional): Save model to this path. Defaults to None.

        Returns:
            float: average training loss
        """
        metric = nlp.load_metric('xnli', experiment_id=4)
        train_loss, best_val_loss = 0.0, float(1e4)
        train_acc = 0.0
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataset) * self._epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_training_steps=total_steps)
        es = EarlyStopping(patience=5)
        for epoch in range(1, self._epochs + 1):
            print(f'Epoch {epoch}/{self._epochs}:')
            kbar = pkbar.Kbar(target=len(train_dataset))
            epoch_acc, epoch_loss = 0.0, 0.0
            self.model.train()
            for batch_idx, sample in enumerate(train_dataset):
                seq = sample["premises_hypotheses"].to(self.device)
                # all tokens that are not set to padding index which is zero
                mask = (seq != 0).to(self.device, dtype=torch.uint8)
                token_types = sample["token_types"].to(self.device)
                labels = sample["outputs"].to(self.device)
                labels_ = labels.view(-1)

                self.optimizer.zero_grad()
                logits = self.model(seq, mask, token_types)
                _, preds = torch.max(logits, dim=-1)
                acc_ = metric.compute(preds, labels_)['accuracy']

                # Comment it out when BERT is frozen
                sample_loss = self.loss_function(logits, labels_)
                sample_loss.backward()

                # Gradient Clipping
                clip_grad_norm_(self.model.parameters(), 1.)

                self.optimizer.step()
                epoch_loss += sample_loss.tolist()
                epoch_acc += acc_.tolist()

                if self._verbose > 0:
                    kbar.update(batch_idx, values=[("loss", sample_loss.item()), ("acc", acc_.item())])

            avg_epoch_loss = epoch_loss / len(train_dataset)
            avg_epoch_acc = epoch_acc / len(train_dataset)
            train_loss += avg_epoch_loss
            train_acc += avg_epoch_acc

            valid_loss, val_acc = self.evaluate(valid_dataset)
            kbar.add(1,
                     values=[("loss", train_loss), ("acc", train_acc), ("val_loss", valid_loss), ("val_acc", val_acc)])
            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', avg_epoch_loss)
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', valid_loss)

            # Update the learning rate.
            # lr_scheduler.step(valid_loss)
            scheduler.step()

            is_best = valid_loss <= best_val_loss
            if is_best:
                logging.info("Model Checkpoint saved")
                best_val_loss = valid_loss
                model_dir = os.path.join(os.getcwd(), 'model',
                                         f'{self.model.name}_ckpt_best')
                self.model.save_(model_dir)

            if es.step(valid_loss):
                print(f"Training Stopped early, epoch #: {epoch}")
                break

        avg_epoch_loss = train_loss / self._epochs
        if save_to is not None:
            self.model.save_(save_to)

        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        """ Evaluation during training process to fetch val_loss & val_acc """
        metric = nlp.load_metric('xnli', experiment_id=153582)
        valid_acc = 0.0
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, desc='Evaluating',
                               leave=False, total=len(valid_dataset)):
                seq = sample["premises_hypotheses"].to(self.device)
                mask = (seq != 0).to(self.device, dtype=torch.uint8)

                labels = sample["outputs"].to(self.device)
                labels_ = labels.view(-1)

                logits = self.model(seq, mask)
                _, preds = torch.max(logits, dim=-1)
                val_acc = metric.compute(preds, labels)['accuracy']
                sample_loss = self.loss_function(logits, labels_)
                valid_loss += sample_loss.tolist()
                valid_acc += val_acc.tolist()
        return valid_loss / len(valid_dataset), valid_acc / len(valid_dataset)

    def save_checkpoint(self, filename):
        """ Saves model's and optimizer's checkpoints to continue training from a certain checkpoint """
        state = {"model": self.model.state_dict(),
                 "optimizer": self.optimizer.state_dict()}
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        """ loads model's and optimizer's checkpoints to continue training from a certain checkpoint """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # now individually transfer the optimizer parts...
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


class XLMTrainer:
    def __init__(self, model, loss_function, optimizer,
                 epochs, verbose, writer, _device):
        """
        Trainer object requires model, criterion (loss function), optimizer
        verbose level and number of epochs
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose
        self._epochs = epochs
        self.writer = writer
        self.device = _device

    def train(self, train_dataset, valid_dataset, save_to=None):
        """
        Trains the model, while keeping track of the best model trained so far
        and saves it if there is an improvement in the validation loss
        Applies Gradients clipping if their norm increased above the 1.0
        Writes the model loss, val_Loss plots in tensorboard.
        Activates early stopping if needed with 5 epochs as patience to prevent overfitting

        It uses pkbar package to have a progress bar keras like,
        the progress bar shows the loss, val_loss during training as well as acc & val_acc
        In order to compute the accuracy, we make use from nlp metric library for XNLI tasks. the accuracy is computed
        from model's predictions during training and the original labels, the predictions are fetched as the argmax of
        the logits tensor.

        Model takes 4 arguments (from the sequences padded as per batch) to do the forward pass:
        1. Languages
        2. Sequence (inputs_ids)
        3. Attention masks
        4. Token types

        Args:
            train_dataset (DataLoader)
            valid_dataset (DataLoader)
            save_to (str, optional): Save model to this path. Defaults to None.

        Returns:
            float: average training loss
            float: average training acc
        """
        metric = nlp.load_metric('xnli')
        train_loss, train_acc, best_val_loss = 0.0, 0.0, float(1e4)
        for epoch in range(1, self._epochs + 1):
            print(f'Epoch {epoch}/{self._epochs}:')
            kbar = pkbar.Kbar(target=len(train_dataset))
            epoch_acc, epoch_loss = 0.0, 0.0
            self.model.train()
            for batch_idx, sample in enumerate(train_dataset):
                seq = sample["premises_hypotheses"].to(self.device)
                mask = sample["attention_mask"].to(self.device)
                token_types = sample["token_types"].to(self.device)
                labels = sample["outputs"].to(self.device)
                labels_ = labels.view(-1)
                languages = torch.LongTensor(sample["languages"]).to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(languages, seq, mask, token_types)
                _, preds = torch.max(logits, dim=-1)
                acc_ = metric.compute(preds, labels_)['accuracy']

                # Comment it out when BERT is frozen
                sample_loss = self.loss_function(logits, labels_)
                sample_loss.backward()

                # Gradient Clipping
                clip_grad_norm_(self.model.parameters(), 1.)

                self.optimizer.step()
                epoch_loss += sample_loss.tolist()
                epoch_acc += acc_.tolist()

                if self._verbose > 0:
                    kbar.update(batch_idx, values=[("loss", sample_loss.item()), ("acc", acc_.item())])

            avg_epoch_loss = epoch_loss / len(train_dataset)
            avg_epoch_acc = epoch_acc / len(train_dataset)
            train_loss += avg_epoch_loss
            train_acc += avg_epoch_acc

            valid_loss, val_acc = self.evaluate(valid_dataset)
            kbar.add(1,
                     values=[("loss", train_loss), ("acc", train_acc), ("val_loss", valid_loss), ("val_acc", val_acc)])
            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', avg_epoch_loss)
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', valid_loss)

            is_best = valid_loss <= best_val_loss
            if is_best:
                logging.info("Model Checkpoint saved")
                best_val_loss = valid_loss
                model_dir = os.path.join(os.getcwd(), 'model',
                                         f'{self.model.name}_ckpt_best')
                self.model.save_(model_dir)
        avg_epoch_loss = train_loss / self._epochs
        avg_epoch_acc = train_acc / self._epochs

        if save_to is not None:
            self.model.save_(save_to)
        return avg_epoch_loss, avg_epoch_acc

    def evaluate(self, valid_dataset):
        """
        Evaluation during training process
        Returns:
            float: average training loss
            float: average training acc
        """
        metric = nlp.load_metric('xnli')
        valid_acc = 0.0
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, desc='Evaluating',
                               leave=False, total=len(valid_dataset)):
                seq = sample["premises_hypotheses"].to(self.device)
                mask = sample["attention_mask"].to(self.device)
                token_types = sample["token_types"].to(self.device)

                labels = sample["outputs"].to(self.device)
                labels_ = labels.view(-1)

                languages = sample["languages"].to(self.device)

                logits = self.model(languages, seq, mask, token_types)
                _, preds = torch.max(logits, dim=-1)
                val_acc = metric.compute(preds, labels)['accuracy']
                sample_loss = self.loss_function(logits, labels_)
                valid_loss += sample_loss.tolist()
                valid_acc += val_acc.tolist()
        return valid_loss / len(valid_dataset), valid_acc / len(valid_dataset)

    def save_checkpoint(self, filename):
        """ Saves model's and optimizer's checkpoints to continue training from a certain checkpoint """
        state = {"model": self.model.state_dict(),
                 "optimizer": self.optimizer.state_dict()}
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        """ loads model's and optimizer's checkpoints to continue training from a certain checkpoint """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # now individually transfer the optimizer parts...
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


class XLMRTrainer:
    def __init__(self, model, loss_function, optimizer,
                 epochs, verbose, writer, _device):
        """
        Trainer object requires model, criterion (loss function), optimizer
        verbose level and number of epochs
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose
        self._epochs = epochs
        self.writer = writer
        self.device = _device

    def train(self, train_dataset, valid_dataset, save_to=None):
        """
        Trains the model, while keeping track of the best model trained so far
        and saves it if there is an improvement in the validation loss
        Applies Gradients clipping if their norm increased above the 1.0
        Writes the model loss, val_Loss plots in tensorboard.
        Activates early stopping if needed with 5 epochs as patience to prevent overfitting

        It uses pkbar package to have a progress bar keras like,
        the progress bar shows the loss, val_loss during training as well as acc & val_acc
        In order to compute the accuracy, we make use from nlp metric library for XNLI tasks. the accuracy is computed
        from model's predictions during training and the original labels, the predictions are fetched as the argmax of
        the logits tensor.

        Model takes 3 arguments (from the sequences padded as per batch) to do the forward pass:
        1. Sequence (inputs_ids)
        2. Attention masks
        3. Token types

        Args:
            train_dataset (DataLoader)
            valid_dataset (DataLoader)
            save_to (str, optional): Save model to this path. Defaults to None.

        Returns:
            float: average training loss
            float: average training acc
        """
        metric = nlp.load_metric('xnli')
        train_loss, train_acc, best_val_loss = 0.0, 0.0, float(1e4)

        for epoch in range(1, self._epochs + 1):
            print(f'Epoch {epoch}/{self._epochs}:')
            kbar = pkbar.Kbar(target=len(train_dataset))

            epoch_acc, epoch_loss = 0.0, 0.0
            self.model.train()
            for batch_idx, sample in enumerate(train_dataset):
                seq = sample["premises_hypotheses"].to(self.device)
                mask = sample["attention_mask"].to(self.device)
                token_types = sample["token_types"].to(self.device)
                labels = sample["outputs"].to(self.device)
                labels_ = labels.view(-1)

                self.optimizer.zero_grad()
                logits = self.model(seq, mask, token_types)
                _, preds = torch.max(logits, dim=-1)
                acc_ = metric.compute(preds, labels_)['accuracy']

                sample_loss = self.loss_function(logits, labels_)
                sample_loss.backward()
                # clip_grad_norm_(self.model.parameters(), 1.)  # Gradient Clipping
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()
                epoch_acc += acc_.tolist()

                if self._verbose > 0:
                    kbar.update(batch_idx, values=[("loss", sample_loss.item()), ("acc", acc_.item())])

            avg_epoch_loss = epoch_loss / len(train_dataset)
            avg_epoch_acc = epoch_acc / len(train_dataset)
            train_loss += avg_epoch_loss
            train_acc += avg_epoch_acc

            valid_loss, val_acc = self.evaluate(valid_dataset)
            kbar.add(1,
                     values=[("loss", train_loss), ("acc", train_acc), ("val_loss", valid_loss), ("val_acc", val_acc)])
            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', avg_epoch_loss)
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', valid_loss)

            is_best = valid_loss <= best_val_loss
            if is_best:
                logging.info("Model Checkpoint saved")
                best_val_loss = valid_loss
                model_dir = os.path.join(os.getcwd(), 'model',
                                         f'{self.model.name}_ckpt_best')
                self.model.save_(model_dir)
        avg_epoch_loss = train_loss / self._epochs
        avg_epoch_acc = train_acc / self._epochs

        if save_to is not None:
            self.model.save_(save_to)
        return avg_epoch_loss, avg_epoch_acc

    def evaluate(self, valid_dataset):
        metric = nlp.load_metric('xnli')
        valid_acc, valid_loss = 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, desc='Evaluating',
                               leave=False, total=len(valid_dataset)):
                seq = sample["premises_hypotheses"].to(self.device)
                mask = sample["attention_mask"].to(self.device)
                token_types = sample["token_types"].to(self.device)
                labels = sample["outputs"].to(self.device)
                labels_ = labels.view(-1)
                logits = self.model(seq, mask, token_types)
                _, preds = torch.max(logits, dim=-1)
                val_acc = metric.compute(preds, labels)['accuracy']
                sample_loss = self.loss_function(logits, labels_)
                valid_loss += sample_loss.tolist()
                valid_acc += val_acc.tolist()
        return valid_loss / len(valid_dataset), valid_acc / len(valid_dataset)

    def save_checkpoint(self, filename):
        """ Saves model's and optimizer's checkpoints to continue training from a certain checkpoint """
        state = {"model": self.model.state_dict(),
                 "optimizer": self.optimizer.state_dict()}
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        """ loads model's and optimizer's checkpoints to continue training from a certain checkpoint """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # now individually transfer the optimizer parts...
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
