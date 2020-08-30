import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent
from transformers import BertModel, XLMModel, AutoModel


class BaselineModel(nn.Module):
    def __init__(self, hparams):
        """
        Creates baseline model which is composed of 2 models (one for premises and the other for hypotheses):
        Embeddings layer > Embeddings dropout > LSTM layer > LSTM dropout > Classifier layer

        Does not freeze the embeddings layer to neither of them, if pretrained embeddings are used then a copy is passed
        to premises embeddings layer and another copy to hypotheses embeddings layer
        :param hparams: HyperParameters
        """
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
        """
        Embeddings layer > Embeddings dropout > LSTM layer > LSTM dropout > Classifier layer
         > [Batch Size, Sequence Length, Embeddings Dim] > [Batch Size, Sequence Length, LSTM Hidden Dim]

        Does not freeze the embeddings layer to neither of them, if pretrained embeddings are used then a copy is passed
        to premises embeddings layer and another copy to hypotheses embeddings layer.

        Premises and hypotheses sequences' each one is passed through its corresponding embeddings layer followed by
        its corresponding LSTMs, Dropout Training is deployed allowing model to get to understand the sentence with
        few words are droped. then outputs of lstm hidden states are concatenated and passed to the classifier layer
        returning the logits
        """
        premises_embeddings = self.premises_word_embedding(premises_seq)
        # [Batch Size, Sequence Length, Embeddings Dim]
        premises_embeddings = self.premises_word_dropout(premises_embeddings)
        premises_lstm_out, (premises_hidden, _) = self.premises_lstm(premises_embeddings)
        # [Batch Size, Sequence Length, LSTM Hidden Dim]
        premises_lstm_out = self.premises_lstm_dropout(premises_lstm_out)

        hypotheses_embeddings = self.hypotheses_word_embedding(hypotheses_seq)
        # [Batch Size, Sequence Length, Embeddings Dim]
        hypotheses_embeddings = self.hypotheses_word_dropout(hypotheses_embeddings)
        hypotheses_lstm_out, (hypotheses_hidden, _) = self.hypotheses_lstm(hypotheses_embeddings)
        # [Batch Size, Sequence Length, LSTM Hidden Dim]
        # hypotheses_lstm ouput = []
        hypotheses_lstm_out = self.hypotheses_lstm_dropout(hypotheses_lstm_out)

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


class K_Model(nn.Module):
    def __init__(self, hparams):
        """
        This model is an extension of the baseline model but with a single Embeddings, LSTM layers instead of 2 like the
        baseline model, this change is done as a consequence of changing the data preprocessing process. Here the data
        is fed as:
         Sentence A [SEP] Sentence B
        The major benefits were on terms of time and number of model parameters
        :param hparams: HyperParameters
        """
        super(K_Model, self).__init__()
        self.name = hparams.model_name
        self.n_hidden = hparams.hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        self.word_dropout = nn.Dropout(hparams.dropout)

        if hparams.embeddings is not None:
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            batch_first=True,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        self.lstm_dropout = nn.Dropout(hparams.dropout)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.classifier = nn.Linear(lstm_output_dim // 2, hparams.num_classes)

    def forward(self, seq):
        """
        Sequences are fed to the model embeddings layer (with Dropout Training) then LSTM + Dropout then classifier layer
        :param seq: [Batch size, max sequence length]
        :return: logits
        """
        embeddings = self.word_embedding(seq)
        embeddings = self.word_dropout(embeddings)
        lstm_out, (hidden, _) = self.lstm(embeddings)
        hidden = self.lstm_dropout(hidden)
        logits = self.classifier(hidden[-1])
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

    def predict_sentence_(self, seq):
        predicted_labels = []
        self.eval()
        with torch.no_grad():
            predictions = self(seq)
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


class BERTModel(nn.Module):
    def __init__(self, hparams, freeze_bert=False):
        """
        Deploying a variant of pretrained language model BERT called mBERT which stands for multilingual_BERT
        BERT is trained with masked language modeling (MLM) objective and next sentence prediction (NSP) on a large
        corpus comprising the Toronto Book Corpus and Wikipedia.
        BERT stands for Bidirectional Encoder Representations from Transformers.

        Here by default I am using 'bert-base-multilingual-uncased' as the model's tokenizer lowers all token
        in the vocabulary.

        If we need to finetune our model, those were the set of parameters advised
            Optimizer: AdamW
            Learning Rate: 2e-5, 3e-5, 5e-5
            Batch Size: 16, 32

        :param hparams: HyperParameters
        :param freeze_bert: Boolean flag whether to freeze all model layers, or to fine tune the model given our task
        """
        super(BERTModel, self).__init__()
        self.name = hparams.model_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-multilingual-uncased')

        if freeze_bert:  # Freeze BERT Layer
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(hparams.dropout)
        classifier_input_dim = self.bert.config.hidden_size
        self.classifier = nn.Linear(classifier_input_dim, hparams.num_classes)

    def forward(self, sequences, attention_mask, tokens_type):
        """
        mBERT takes 3 args as explained below, then the
        mBERT forward method takes
        :param sequences: (inputs_ids) Indices of input sequence tokens in the vocabulary.
        :param attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        :param tokens_type: Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in [0, 1]: 0 corresponds to a sentence A token, 1 corresponds to a sentence B token

        mBERT Output is
            - last_hidden_state [batch_size, sequence_length, hidden_size] which represents the Sequence
                of hidden-states at the output of the last layer of the model.
            - pooled_output [batch_size, hidden_size]: Last layer hidden-state of the first token of the
                sequence [CLS], usually not a good summary of the semantic content of the input, as per documentation

        :return: logits
            Which is produced after taking the mean of bert last_hidden_layer then passed to classifier layer
        """
        bert_hidden_layer, pooled_output = self.bert(input_ids=sequences,
                                                     attention_mask=attention_mask,
                                                     token_type_ids=tokens_type)
        sentence_embeddings = torch.mean(bert_hidden_layer, dim=1)
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


class XLM_Model(nn.Module):
    def __init__(self, hparams, model_name="xlm-mlm-tlm-xnli15-1024", freeze_model=False):
        """
        Deploying a Cross-lingual Language Model.
        It’s a transformer pre-trained using one of the following objectives:
            1. CLM (Casual Language Modeling)
            2. MLM (Masked language modeling)
            3. TLM (Translated language modeling)
        Here we are using the model trained with MLM and TLM training objectives as it yielded better results
        than the model trained only with MLM objective.

        :param hparams: hyperparameters
        :param model_name: by default it is set to "xlm-mlm-tlm-xnli15-1024" which is the XLM model trained with both
        MLM and TLM objectives and fine tuned for XNLI task (15 languages) with a hidden dim of 1024
        :param freeze_bert: Boolean flag whether to freeze all model layers, or to fine tune the model given our task
        """
        super(XLM_Model, self).__init__()
        self.name = hparams.model_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_model = XLMModel.from_pretrained(model_name,
                                                         output_hidden_states=True,
                                                         output_attentions=False)

        if freeze_model:  # Freeze model Layer
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(hparams.dropout)
        classifier_input_dim = self.pretrained_model.config.hidden_size
        self.classifier = nn.Linear(classifier_input_dim, hparams.num_classes)

    def forward(self, languages, sequences, attention_mask, tokens_type):
        """
        XLM takes 4 args as explained below.
         Forward method takes

        :param languages: A parallel sequence of tokens used to indicate the language of each token in the input seq.
        :param sequences: (inputs_ids) Indices of input sequence tokens in the vocabulary.
        :param attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        :param tokens_type: Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in [0, 1]: 0 corresponds to a sentence A token, 1 corresponds to a sentence B token

        XLM Output is
            - last_hidden_state [batch_size, sequence_length, hidden_size] which represents the Sequence
                of hidden-states at the output of the last layer of the model.
            - other parameters that were not used here but can be found in the documentation of transformers package

        :return: logits
            Which is produced after taking the mean of XLM last_hidden_layer then passed to classifier layer
        """
        outputs = self.pretrained_model(langs=languages,
                                        input_ids=sequences,
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

    def predict_sentence_(self, languages, seq, mask, tokens_type):
        predicted_labels = []
        self.eval()
        with torch.no_grad():
            predictions = self(languages, seq, mask, tokens_type)
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


class XLMR_Model(nn.Module):
    def __init__(self, hparams, model_name, freeze_model=False):
        """
        Cross-lingual Language Model which is based on RoBERTa. It is a large multi-lingual language model,
        trained on 2.5TB of filtered CommonCrawl data. It is a transformer-based masked language model on
        100 languages.

        RoBERTa is built on BERT, but, it modifies key hyperparameters, removes BERT's NSP training objective,
        and train with larger learning rates, and larger batch sizes.

        :param hparams: hyperparameters
        :param model_name: by default it is set to "xlm-roberta-large"
        :param freeze_bert: Boolean flag whether to freeze all model layers, or to fine tune the model given our task
        """
        super(XLMR_Model, self).__init__()
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
        """
        XLM-RoBERTa is unlike XLM as it does not require `lang` tensors to understand which language
        is used, and should be able to determine the correct language from the input ids.
        XLM-RoBERTa takes 3 args as explained below.
         Forward method takes
        :param sequences: (inputs_ids) Indices of input sequence tokens in the vocabulary.
        :param attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        :param tokens_type: Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in [0, 1]: 0 corresponds to a sentence A token, 1 corresponds to a sentence B token

        XLM Output is
            - last_hidden_state [batch_size, sequence_length, hidden_size] which represents the Sequence
                of hidden-states at the output of the last layer of the model.
            - other parameters that were not used here but can be found in the documentation of transformers package

        :return: logits
            Which is produced after taking the mean of XLM last_hidden_layer then passed to classifier layer

        Note for tokenizer:
        it is unlike bert's tokenizer it uses BPE as a tokenizer and a different pre-training scheme.

        """

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
        state_dict = torch.load(path) if self._device == 'cuda' else torch.load(path, map_location=self._device)
        self.load_state_dict(state_dict)

    def predict_sentence_(self, seq, mask, tokens_type):
        self.eval()
        with torch.no_grad():
            predictions = self(seq, mask, tokens_type)
            _, argmax = torch.max(predictions, dim=-1)
        return argmax.tolist()

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
