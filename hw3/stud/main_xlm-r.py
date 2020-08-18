import json
import os
import pickle
import datetime
import random
import numpy as np
from pathlib import Path
import logging
import importlib
import warnings
import time

import nlp
import nltk
import torch
from tabulate import tabulate
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import torch.nn as nn
from torch.nn.modules.module import _addindent
from nltk.tokenize import RegexpTokenizer

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (AutoTokenizer, AutoModel,
                          AdamW, get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

import pkbar
import shutil


def manage_workspace_directories():
    dir_path = os.getcwd()
    folders = ['model', 'runs']
    folders_to_remove = ['sample_data']

    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(os.path.join(dir_path, folder))

    for folder in folders_to_remove:
        if os.path.exists(folder):
            shutil.rmtree(folder)


def configure_workspace(SEED=1873337):
    """
    Configure seed for reproducability, configure logging messages
    """
    warnings.filterwarnings("ignore")
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nltk.download('punkt', quiet=True)
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.WARNING)
    manage_workspace_directories()


def save_pickle(save_to, save_what):
    """ Save data into pickle format"""
    with open(save_to, mode='wb') as f:
        pickle.dump(save_what, f)


def load_pickle(load_from):
    """ Load data from presaved pickle file"""
    with open(load_from, 'rb') as f:
        return pickle.load(f)


def ensure_dir(path):
    """
    Makes sure direcctory exists and if not it creates it
    """
    if not os.path.exists(path):
        os.makedirs(path)


"""## Dataset Parser"""


class PTMDatasetParser(Dataset):
    """ PreTrainedModel (XLM) DatasetParser """

    def __init__(self, _device, data_, model_="bert-base-multilingual-uncased"):
        super(PTMDatasetParser).__init__()
        self.encoded_data = []
        self.device = _device
        self.languages, self.premises, self.hypotheses, self.labels = data_
        self.model_name = model_
        self.label2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.idx2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Dataset is not indexed yet.\
                                To fetch raw elements, use get_element(idx)")
        return self.encoded_data[idx]

    def get_element(self, idx):
        return self.languages[idx], self.premises[idx], self.hypotheses[idx], self.labels[idx]

    def encode_dataset(self):
        """ Indexing and encoding the dataset """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=True)

        for lang_, premise_, hypothesis_, labels_ in tqdm(
                zip(self.languages, self.premises, self.hypotheses, self.labels),
                leave=False, total=len(self.premises),
                desc=f'Encoding dataset'):
            encoded_dict = tokenizer.encode_plus(text=premise_,
                                                 text_pair=hypothesis_,
                                                 add_special_tokens=True,
                                                 return_token_type_ids=True,
                                                 return_attention_mask=True,
                                                 max_length=128,
                                                 truncation=True,
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
            self.encoded_data.append({'premises_hypotheses': torch.squeeze(encoded_dict["input_ids"]),
                                      'attention_mask': torch.squeeze(encoded_dict["attention_mask"]),
                                      'token_types': torch.squeeze(encoded_dict["token_type_ids"]),
                                      'outputs': torch.LongTensor([self.label2idx.get(labels_)])})

    @staticmethod
    def pad_batch(batch):
        """ Pads sequences per batch with `padding_value=0`
        Args: batch: List[dict]
        Returns: dict of models inputs padded as per max len in batch
        """
        # ('languages', 'premises_hypotheses', 'token_types','outputs')
        premises_hypotheses_batch = [sample["premises_hypotheses"] for sample in batch]
        padded_premises_hypotheses = pad_sequence(premises_hypotheses_batch, padding_value=2, batch_first=True)

        mask = [sample["attention_mask"] for sample in batch]
        padded_mask = pad_sequence(mask, padding_value=0, batch_first=True)

        # Token types is padded with 1 (unlike others padded with 0), padded with 1's due to
        # that we are doing pair_sentences encoding so the other sentence is consisting of 1's
        token_types_batch = [sample["token_types"] for sample in batch]
        padded_token_types = pad_sequence(token_types_batch, padding_value=1, batch_first=True)
        outputs_batch = pad_sequence([sample["outputs"] for sample in batch], batch_first=True)

        return {"premises_hypotheses": padded_premises_hypotheses,
                "attention_mask": padded_mask,
                "token_types": padded_token_types,
                "outputs": outputs_batch}

    def decode_predictions(self, predictions):
        """
        Flattens predictions list (if it is a list of lists)
        and get the corresponding label name for each label index (label_stoi)
        """
        if any(isinstance(el, list) for el in predictions):
            return [self.idx2label.get(label) for tag in predictions for label in tag]
        else:
            predictions_ = [_e for e in predictions for _e in e]
            return [self.idx2label.get(label) for tag in predictions_ for label in tag]


int2nli_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
nli_label2int = {v: k for k, v in int2nli_label.items()}
nli_labels = list(nli_label2int.keys())


def read_mnli(dataset):
    """
    READ MNLI Dataset returns 4 lists each of them is a list of strings
    languages, premises, hypotheses, labels
    """
    int2nli_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    languages = []
    premises = []
    hypotheses = []
    labels = []

    for sample in dataset:
        languages.append('en')
        premises.append(sample['premise'])
        hypotheses.append(sample['hypothesis'])
        labels.append(int2nli_label[sample['label']])

    assert len(languages) == len(premises) == len(hypotheses) == len(labels)
    return languages, premises, hypotheses, labels


def read_xnli(dataset):
    languages = []
    premises = []
    hypotheses = []
    labels = []

    for sample in dataset:

        language2premise = {}
        language2hypothesis = {}

        # read premises
        for language, premise in sample['premise'].items():
            language2premise[language] = premise

        # read hypotheses
        for language, hypothesis in zip(sample['hypothesis']['language'], sample['hypothesis']['translation']):
            language2hypothesis[language] = hypothesis

        label = int2nli_label[sample['label']]

        sample_languages = set(language2premise.keys()) & set(language2hypothesis.keys())
        assert len(sample_languages) == len(language2premise) == len(language2hypothesis)

        for language in sample_languages:
            languages.append(language)
            premises.append(language2premise[language])
            hypotheses.append(language2hypothesis[language])
            labels.append(label)

    assert len(languages) == len(premises) == len(hypotheses) == len(labels)
    return languages, premises, hypotheses, labels


def read_train():
    """
    Read training dataset (MNLI)
    """
    return read_mnli(nlp.load_dataset('multi_nli')['train'])


def read_dev():
    """
    Read validation dataset (MNLI)
    """
    return read_mnli(nlp.load_dataset('multi_nli')['validation_matched'])


def read_test():
    """
    Read test dataset (XNLI)
    """
    return read_xnli(nlp.load_dataset('xnli')['test'])


"""## Models"""


class PTM_Model(nn.Module):
    def __init__(self, hparams, model_name, freeze_model=False):
        super(PTM_Model, self).__init__()
        self.name = hparams.model_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_model = AutoModel.from_pretrained(model_name)

        if freeze_model:  # Freeze model Layer
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(hparams.dropout)
        classifier_input_dim = self.pretrained_model.config.hidden_size
        self.classifier = nn.Linear(classifier_input_dim, hparams.num_classes)

    def forward(self, sequences, attention_mask, tokens_type):
        outputs = self.pretrained_model(input_ids=sequences,
                                        attention_mask=attention_mask,
                                        token_type_ids=tokens_type)
        last_hidden_state = outputs[0]
        sentence_embeddings = torch.mean(last_hidden_state, dim=1)
        o = self.dropout(sentence_embeddings)
        logits = self.classifier(o)
        return logits

    def save_(self, dir_path):
        """
        Saves model and its state dict into the given dir path
        Args: dir_path (str)
        """
        torch.save(self, f'{dir_path}.pt')
        torch.save(self.state_dict(), f'{dir_path}.pth')

    def load_(self, path):
        """
        Loads the model and its state dictionary
        Args: path (str): [Model's state dict is located]
        """
        state_dict = torch.load(path) if self.device == 'cuda' else torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def predict_sentence_(self, seq, mask, tokens_type):
        predicted_labels = []
        self.eval()
        with torch.no_grad():
            predictions = self(seq, mask, tokens_type)
            _, argmax = torch.max(predictions, dim=-1)
            predicted_labels.append(argmax.tolist())
        return predicted_labels

    def print_summary(self, show_weights=False, show_parameters=False):
        """
        Summarizes torch model by showing trainable parameters and weights.
        """
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self._modules.items():
            # if it contains layers let call it recursively to get params and weights
            if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential
            ]:
                modstr = self.print_summary()
            else:
                modstr = module.__repr__()
            modstr = _addindent(modstr, 2)

            params = sum([np.prod(p.size()) for p in module.parameters()])
            weights = tuple([tuple(p.size()) for p in module.parameters()])

            tmpstr += '  (' + key + '): ' + modstr
            if show_weights:
                tmpstr += ', weights={}'.format(weights)
            if show_parameters:
                tmpstr += ', parameters={}'.format(params)
            tmpstr += '\n'

        tmpstr = tmpstr + ')'
        print(f'========== {self.name} Model Summary ==========')
        print(tmpstr)
        num_params = sum(p.numel()
                         for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params:,}")
        print('==================================================')


"""### Hyperparameters"""


class HyperParameters:
    """
    Hyperparameters configuration class where all vars are defined
    """

    def __init__(self, model_name_, vocab, label_vocab, embeddings_, batch_size_):
        self.model_name = model_name_
        self.vocab_size = len(vocab) if vocab else "Using AutoTokenizer's vocab"
        self.num_classes = len(label_vocab)
        self.hidden_dim = 128
        self.bidirectional = True
        self.embedding_dim = 300
        self.num_layers = 1
        self.dropout = 0.3
        self.embeddings = embeddings_
        self.batch_size = batch_size_

    def _print_info(self):
        """
        prints summary of model's hyperparameters
        """
        print("========== Hyperparameters ==========",
              f"Name: {self.model_name.replace('_', ' ')}",
              f"Vocab Size: {self.vocab_size}",
              f"Tags Size: {self.num_classes}",
              f"Embeddings Dim: {self.embedding_dim}",
              f"Hidden Size: {self.hidden_dim}",
              f"BiLSTM: {self.bidirectional}",
              f"Layers Num: {self.num_layers}",
              f"Dropout: {self.dropout}",
              f"Pretrained_embeddings: {False if self.embeddings is None else True}",
              f"Batch Size: {self.batch_size}", sep='\n')


"""## Trainer"""

from random import randint


class PTMTrainer:
    def __init__(self, model, loss_function, optimizer,
                 epochs, verbose, writer, _device):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose
        self._epochs = epochs
        self.writer = writer
        self.device = _device

    def train(self, train_dataset, valid_dataset, save_to=None):
        metric = nlp.load_metric('xnli', experiment_id=randint(0, 50))
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
                    tok = time.time()
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
        metric = nlp.load_metric('xnli', experiment_id=randint(0, 50))
        valid_acc, valid_loss = 0.0, 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, desc='Evaluating',
                               leave=False, total=len(valid_dataset)):
                seq = sample["premises_hypotheses"].to(self.device)
                # mask = (seq != 2).to(self.device, dtype=torch.uint8)
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
        state = {"model": self.model.state_dict(),
                 "optimizer": self.optimizer.state_dict()}
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # now individually transfer the optimizer parts...
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


"""### Callbacks"""


class WriterTensorboardX:
    """
    Logs training process to tensorboard for visualization
    """

    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        ensure_dir(writer_dir)
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                os.system('pip install tensorboardX')
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text',
                                        'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)

            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr


"""## Evaluate Performance"""


def compute_metrics(languages, predicted_labels, labels):
    """
    Computes accuracy of the predicted labels (y_hat) vs labels (y)
    """
    labels = [nli_label2int[label] for label in labels]
    predicted_labels = [nli_label2int[predicted_label] for predicted_label in predicted_labels]

    metric = nlp.load_metric('xnli', experiment_id=101)
    headers = ('', 'accuracy', '# samples')
    table = []
    table.append(('overall', metric.compute(predicted_labels, labels)['accuracy'], len(labels)))

    # per language
    for evaluation_language in sorted(set(languages)):
        evaluation_language_predicted_labels = [predicted_label for predicted_label, language in
                                                zip(predicted_labels, languages) if language == evaluation_language]
        evaluation_language_labels = [label for label, language in zip(labels, languages) if
                                      language == evaluation_language]
        evaluation_language_accuracy = metric.compute(evaluation_language_predicted_labels, evaluation_language_labels)[
            'accuracy']
        table.append((evaluation_language, evaluation_language_accuracy, len(evaluation_language_labels)))
    print(tabulate(table, headers=headers, tablefmt="pretty"))


""" Parse and pickle dataset """


def print_summary(summary_data, verbose=False):
    premises, dev_premises, test_premises, word2idx, label2idx = summary_data
    # To clear out cell from unwanted downloading and indexing progress bars
    if verbose:
        print("\n=============Data Summary======================",
              f"train_x length: {len(premises)} sentences",
              f"dev_x length: {len(dev_premises)} sentences",
              f"test_x length: {len(test_premises)} sentences",
              f"Vocab size: {len(word2idx)}",
              f"Labels vocab size: {len(label2idx)}",
              "===============================================\n", sep="\n")


if __name__ == "__main__":
    configure_workspace()
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "model", "word_stoi.pkl")
    model_name_ = "xlm-roberta-large"

    data = read_train()
    train_dataset = PTMDatasetParser(device_, data, model_name_)
    train_dataset.encode_dataset()
    print("Processed and encoded Training dataset")

    dev_data = read_dev()
    dev_dataset = PTMDatasetParser(device_, dev_data, model_name_)
    dev_dataset.encode_dataset()
    print("Processed and encoded Validation dataset")

    test_data = read_test()
    test_dataset = PTMDatasetParser(device_, test_data, model_name_)
    test_dataset.encode_dataset()
    print("Processed and encoded Testing dataset")

    """ Hyperparameters & DataLoaders """

    # Set Hyperparameters
    pretrained_embeddings_ = None
    batch_size = 8

    name_ = 'XLM-R_Seq_CLS'
    hp = HyperParameters(name_, None, train_dataset.label2idx,
                         pretrained_embeddings_, batch_size)
    hp._print_info()

    # Prepare data loaders
    train_dataset_ = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                collate_fn=PTMDatasetParser.pad_batch,
                                shuffle=True)
    dev_dataset_ = DataLoader(dataset=dev_dataset,
                              batch_size=batch_size,
                              collate_fn=PTMDatasetParser.pad_batch)
    test_dataset_ = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               collate_fn=PTMDatasetParser.pad_batch)

    """ Build and train model"""

    model = PTM_Model(hp, model_name_).to(device_)
    model.print_summary()

    log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
    writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)
    optimizer_ = AdamW(model.parameters(), lr=5e-6)
    epochs_ = 1

    trainer = PTMTrainer(model=model, writer=writer_,
                         epochs=epochs_, _device=device_,
                         loss_function=CrossEntropyLoss(),
                         optimizer=optimizer_, verbose=True)

    save_to_ = os.path.join(os.getcwd(), 'model', f"{model.name}_model")

    try:
        _, _ = trainer.train(train_dataset_, dev_dataset_, save_to=save_to_)
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\nModel's & Optimizer's states were saved before halting due to {e}")
        model.save_(save_to_)
        trainer.save_checkpoint(filename=f"{model.name}_ckpt_retrain.pt")


    """ Predict and compute scores """

    predicted_labels = []
    for sample in tqdm(test_dataset_, desc='Predicting on Testing dataset',
                       leave=False, total=len(test_dataset_)):
        seq = sample["premises_hypotheses"].to(device_)
        mask = sample["attention_mask"].to(device_)
        labels = sample["outputs"].to(device_)
        token_types = sample["token_types"].to(device_)
        batch_predicted_labels = model.predict_sentence_(seq, mask, token_types)
        predicted_labels.extend(batch_predicted_labels)
    # save_pickle(os.path.join(os.getcwd(), "model", "predicted_labels.pkl"), predicted_labels)

    decoded_predicted_labels = [train_dataset.idx2label.get(tag_) for tag in predicted_labels for tag_ in tag]
    assert len(decoded_predicted_labels) == len(test_dataset.labels)
    compute_metrics(test_dataset.languages, decoded_predicted_labels, test_dataset.labels)
