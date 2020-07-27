import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent


class BaselineModel(nn.Module):
    def __init__(self, hparams):
        super(BaselineModel, self).__init__()
        self.name = hparams.model_name
        self.n_hidden = hparams.hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.premises_word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        self.premises_word_dropout = nn.Dropout(hparams.dropout)

        self.hypotheses_word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        self.hypotheses_word_dropout = nn.Dropout(hparams.dropout)

        if hparams.embeddings is not None:
            self.premises_word_embedding.weight.data.copy_(hparams.embeddings)
            self.hypotheses_word_embedding.weight.data.copy_(hparams.embeddings)
            self.premises_word_embedding.weight.requires_grad = False
            self.hypotheses_word_embedding.weight.requires_grad = False

        self.premises_lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                                     bidirectional=hparams.bidirectional,
                                     num_layers=hparams.num_layers,
                                     batch_first=True,
                                     dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        self.premises_lstm_dropout = nn.Dropout(hparams.dropout)

        self.hypotheses_lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                                       bidirectional=hparams.bidirectional,
                                       num_layers=hparams.num_layers,
                                       batch_first=True,
                                       dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        self.hypotheses_lstm_dropout = nn.Dropout(hparams.dropout)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    def forward(self, premises_seq, hypotheses_seq):
        premises_embeddings = self.premises_word_embedding(premises_seq)
        # premises_embeddings = self.premises_word_dropout(premises_embeddings)
        premises_lstm_out, (premises_hidden, _) = self.premises_lstm(premises_embeddings)
        # premises_lstm_out = self.premises_lstm_dropout(premises_lstm_out)

        hypotheses_embeddings = self.hypotheses_word_embedding(hypotheses_seq)
        # hypotheses_embeddings = self.hypotheses_word_dropout(hypotheses_embeddings)
        hypotheses_lstm_out, (hypotheses_hidden, _) = self.hypotheses_lstm(hypotheses_embeddings)
        # hypotheses_lstm_out = self.hypotheses_lstm_dropout(hypotheses_lstm_out)

        concat_lstm_hidden = torch.cat((premises_hidden[-1], premises_hidden[-1]), dim=-1)
        logits = self.classifier(concat_lstm_hidden)
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

    def predict_sentence_(self, premises_seq, hypotheses_seq):
        predicted_labels = []
        self.eval()
        with torch.no_grad():
            predictions = self(premises_seq, hypotheses_seq)
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