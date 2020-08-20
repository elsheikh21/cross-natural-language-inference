import collections
import os
import random
import string
from pathlib import Path
from random import random

import matplotlib as mlp
import matplotlib.pyplot as plt
import nlp
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from nltk.tokenize import RegexpTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, XLMTokenizer
from utils import load_pickle, save_pickle
from wordcloud import WordCloud, STOPWORDS

nltk.download('stopwords')

int2nli_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
nli_label2int = {v: k for k, v in int2nli_label.items()}
nli_labels = list(nli_label2int.keys())


def read_test():
    """ Read test dataset (XNLI) """
    return read_xnli(nlp.load_dataset('xnli')['test'])


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


def read_train():
    """ Read training dataset (MNLI) """
    return read_mnli(nlp.load_dataset('multi_nli')['train'])


def read_dev():
    """ Read validation dataset (MNLI) """
    return read_mnli(nlp.load_dataset('multi_nli')['validation_matched'])


class NLPDatasetParser(Dataset):
    def __init__(self, _device, data):
        """
        Takes NLP Dataset (from Hugging Face NLP package),
        creates vocabulary, indexes dataset, pad per batch

        Args:
            data [List[str], List[str], List[str], List[str]]: A tuple of 4 lists containing
            language, premise, hypothesis, and labels
        """
        super(NLPDatasetParser).__init__()
        self.encoded_data = []
        self.device = _device
        self.languages, self.premises, self.hypotheses, self.labels = data

    def create_vocabulary(self, load_from=None):
        """
        Creates vocab dictionaries for both data_x which is composed of 2 lists (premises and hypotheses), and
        creates labels vocab dictionary. Unless the path given `load_from` has a file to unpickle vocabulary from it
        Auxiliary functionality: saves the vocab dict

        Args:
            load_from (str, optional): Path to save the vocab dict. Defaults to None.

        Returns:
            stoi, itos,labels_stoi, labels_itos: vocabulary dictionaries
        """
        if load_from is not None and Path(load_from).is_file():
            stoi = load_pickle(load_from)
            itos = {key: val for key, val in enumerate(stoi)}
            labels_itos = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            labels_stoi = {v: k for k, v in labels_itos.items()}
            return stoi, itos, labels_stoi, labels_itos
        # Using RegexpTokenizer took less time than nltk.word_tokenizer() and yielded better vocabulary
        # in terms of less number of redundant tokens
        tokenizer = RegexpTokenizer("[\w]+")
        premises_words = [word for word in tokenizer.tokenize(" ".join(self.premises))]
        hypotheses_words = [word for word in tokenizer.tokenize(" ".join(self.hypotheses))]
        premises_words.extend(hypotheses_words)
        unigrams = sorted(list(set(premises_words)))
        stoi = {'[PAD]': 0, '[UNK]': 1}
        start_ = 2
        stoi.update({val: key for key, val in enumerate(unigrams, start=start_)})
        itos = {key: val for key, val in enumerate(stoi)}
        if load_from is not None:
            save_pickle(load_from, stoi)
            save_pickle(load_from.replace('stoi', 'itos'), itos)
        labels_itos = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        labels_stoi = {v: k for k, v in labels_itos.items()}
        return stoi, itos, labels_stoi, labels_itos

    def encode_dataset(self, word2idx, label2idx):
        """
        Indexing and encoding the dataset

        Args:
            word2idx (dict): Vocab dictionary
            label2idx (dict): Labels dictionary
        """
        for lang_, premise_, hypothesis_, labels in tqdm(
                zip(self.languages, self.premises, self.hypotheses, self.labels),
                leave=False, total=len(self.premises),
                desc=f'Indexing dataset to {self.device}'):
            self.encoded_data.append({'languages': lang_,
                                      'premises': torch.LongTensor([word2idx.get(word, 1) for word in premise_]),
                                      'hypotheses': torch.LongTensor([word2idx.get(word, 1) for word in hypothesis_]),
                                      'outputs': torch.LongTensor([label2idx.get(labels)])})

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Dataset is not indexed yet. To fetch raw elements, use get_element(idx)")
        return self.encoded_data[idx]

    def get_element(self, idx):
        return self.languages[idx], self.premises[idx], self.hypotheses[idx], self.labels[idx]

    @staticmethod
    def pad_batch(batch):
        """
        Pads sequences per batch with `padding_value=0`

        Args:
            batch: List[dict]

        Returns:
            dict of models inputs padded as per max len in batch
        """
        # A naive workaround in order to either provide the required model with either paired_sequences
        # or with single sequences
        premises_hypotheses_flag = True
        try:
            _ = [sample["premises_hypotheses"] for sample in batch]
        except KeyError:
            premises_hypotheses_flag = False

        languages_batch = [sample["languages"] for sample in batch]
        outputs_batch = pad_sequence([sample["outputs"] for sample in batch], batch_first=True)

        if not premises_hypotheses_flag:
            premises_batch = [sample["premises"] for sample in batch]
            padded_premise = pad_sequence(premises_batch, batch_first=True)

            # Hypotheses list of tensors has to be of the same size as
            # premises for the model to function properly
            hypotheses_batch = [sample["hypotheses"] for sample in batch]
            # For that to happen -hypothesis batch to be the same length
            # as the premises- we had to append the last element from padded_premise
            hypotheses_batch.append(padded_premise[-1])
            # & then remove it from the padded_hypotheses before feeding it to the model
            padded_hypotheses = pad_sequence(hypotheses_batch, batch_first=True)[:-1]

            return {'languages': languages_batch,
                    'premises': padded_premise,
                    'hypotheses': padded_hypotheses,
                    'outputs': outputs_batch}
        else:
            premises_hypotheses_batch = [sample["premises_hypotheses"] for sample in batch]
            padded_premises_hypotheses = pad_sequence(premises_hypotheses_batch, batch_first=True)

            return {'languages': languages_batch,
                    'premises_hypotheses': padded_premises_hypotheses,
                    'outputs': outputs_batch}

    @staticmethod
    def decode_predictions(predictions, label_itos):
        """
        Flattens predictions list (if it is a list of lists)
        and get the corresponding label name for each label index (label_stoi)
        """
        if any(isinstance(el, list) for el in predictions):
            return [label_itos.get(label) for tag in predictions for label in tag]
        else:
            predictions_ = [_e for e in predictions for _e in e]
            return [label_itos.get(label) for tag in predictions_ for label in tag]


class K_NLPDatasetParser(NLPDatasetParser):
    def __init__(self, _device, data_, is_bert):
        """
        Takes NLP Dataset (from Hugging Face NLP package),
        creates vocabulary, index dataset, pad per batch

        Args:
            data [List[str], List[str], List[str], List[str]]: A tuple of 4 lists containing
            language, premise, hypothesis, and labels
        """
        super(K_NLPDatasetParser).__init__()
        self.encoded_data, self.premises_hypotheses = [], []
        self.languages, self.premises, self.hypotheses, self.labels = data_
        self.device = _device
        self.use_bert = is_bert
        self.process_dataset()

    def create_vocabulary(self, load_from=None):
        """
        Creates vocab dictionaries for both data_x which is composed of 2 lists (premises and hypotheses), and
        creates labels vocab dictionary. Unless the path given `load_from` has a file to unpickle vocabulary from it
        Auxiliary functionality: saves the vocab dict

        Args: load_from (str, optional): Path to save the vocab dict. Defaults to None.
        Returns: stoi, itos,labels_stoi, labels_itos: vocabulary dictionaries
        """
        if load_from is not None and Path(load_from).is_file():
            stoi = load_pickle(load_from)
            itos = {key: val for key, val in enumerate(stoi)}
            labels_itos = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            labels_stoi = {v: k for k, v in labels_itos.items()}
            return stoi, itos, labels_stoi, labels_itos
        # Using RegexpTokenizer took less time than nltk.word_tokenizer() and yielded better vocabulary
        # in terms of less number of redundant tokens
        tokenizer = RegexpTokenizer("[\w]+")
        words = [word for word in tokenizer.tokenize(" ".join(self.premises_hypotheses))]
        unigrams = sorted(list(set(words)))
        stoi = {'[PAD]': 0, '[UNK]': 1, '[SEP]': 2}
        start_ = 3
        stoi.update({val: key for key, val in enumerate(unigrams, start=start_)})
        itos = {key: val for key, val in enumerate(stoi)}
        if load_from is not None:
            save_pickle(load_from, stoi)
            save_pickle(load_from.replace('stoi', 'itos'), itos)
        labels_itos = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        labels_stoi = {v: k for k, v in labels_itos.items()}
        return stoi, itos, labels_stoi, labels_itos

    def encode_dataset(self, word2idx, label2idx):
        """ Indexing and encoding the dataset
        In case of using BERT Tokenizer, I used `encode_plus()` which takes
            - premise_hypothesis_ for text argument
        and it adds to it special tokens [CLS] & [SEP] indicating the first token of a sentence and
        separator between sentences or pair of sequences, respectively. It returns as well the token type ids
        which allows the model to differentiate between sentence A and sentence B, as well as, attention masks to
        help model distinguish between [PAD] tokens and original sentence tokens

        Args:
            word2idx (dict): Vocab dictionary
            label2idx (dict): Labels dictionary
        """
        lang_x, premises_hypotheses_x_stoi, attention_masks, data_y_stoi = [], [], [], []
        if self.use_bert:
            model_name = "bert-base-multilingual-cased"
            bert_tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
            for lang_, premise_hypothesis_, labels_ in tqdm(zip(self.languages, self.premises_hypotheses, self.labels),
                                                            leave=False, total=len(self.premises_hypotheses),
                                                            desc=f'Indexing dataset to {self.device} using mBERT tokenizer'):
                lang_x.append(lang_)
                encoding = bert_tokenizer.encode_plus(
                    premise_hypothesis_,
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    return_tensors='pt'  # Return PyTorch tensors
                )
                premises_hypotheses_x_stoi.append(torch.squeeze(encoding['input_ids']))
                attention_masks.append(torch.squeeze(encoding['attention_mask']))
                data_y_stoi.append(torch.LongTensor([label2idx.get(labels_)]))
        else:
            for lang_, premise_hypothesis_, labels in tqdm(zip(self.languages, self.premises_hypotheses, self.labels),
                                                           leave=False, total=len(self.premises_hypotheses),
                                                           desc=f'Indexing dataset to {self.device}'):
                lang_x.append(lang_)
                premises_hypotheses_x_stoi.append(
                    torch.LongTensor([word2idx.get(word, 1) for word in premise_hypothesis_]))
                data_y_stoi.append(torch.LongTensor([label2idx.get(labels)]))
        if len(attention_masks) > 0:
            for i in tqdm(range(len(premises_hypotheses_x_stoi)), desc="Encoding dataset",
                          leave=False, total=len(self.premises_hypotheses)):
                self.encoded_data.append({'languages': lang_x[i],
                                          'premises_hypotheses': premises_hypotheses_x_stoi[i],
                                          'attention_masks': attention_masks[i],
                                          'outputs': data_y_stoi[i]})
        else:
            for i in tqdm(range(len(premises_hypotheses_x_stoi)), desc="Encoding dataset",
                          leave=False, total=len(self.premises_hypotheses)):
                self.encoded_data.append({'languages': lang_x[i],
                                          'premises_hypotheses': premises_hypotheses_x_stoi[i],
                                          'outputs': data_y_stoi[i]})

    def __len__(self):
        return len(self.premises_hypotheses)

    def get_element(self, idx):
        return self.languages[idx], self.premises_hypotheses[idx], self.labels[idx]

    @staticmethod
    def pad_batch(batch):
        """
        Pads sequences per batch with `padding_value=0` in order to feed model tensors of the same sequence length
        Args: batch: List[dict]
        Returns: dict of models inputs padded as per max len in batch
        """
        languages_batch = [sample["languages"] for sample in batch]
        premises_hypotheses_batch = [sample["premises_hypotheses"] for sample in batch]
        padded_premises_hypotheses = pad_sequence(premises_hypotheses_batch, batch_first=True)
        outputs_batch = pad_sequence([sample["outputs"] for sample in batch], batch_first=True)

        return {'languages': languages_batch,
                'premises_hypotheses': padded_premises_hypotheses,
                'outputs': outputs_batch}

    def process_dataset(self):
        """
        To merge premises with their corresponding hypotheses sentences with a [SEP] token following
        a simple form of BERT pair encoding. And then deleting both lists premises and hypotheses
        """
        for premises_, hypotheses_ in tqdm(zip(self.premises, self.hypotheses), leave=False,
                                           desc='Processing Dataset', total=len(self.premises)):
            self.premises_hypotheses.append(f"{premises_.strip()} [SEP] {hypotheses_.strip()}")
        del self.premises, self.hypotheses


class BERTDatasetParser(Dataset):
    """
    Takes NLP Dataset (from Hugging Face NLP package), uses the bert_tokenizer vocabulary
    creates labels vocabulary, index dataset, pad per batch

    Args:
        data [List[str], List[str], List[str], List[str]]: A tuple of 4 lists containing
        language, premise, hypothesis, and labels
    """

    def __init__(self, _device, data_):
        super(BERTDatasetParser).__init__()
        self.encoded_data = []
        self.device = _device
        self.languages, self.premises, self.hypotheses, self.labels = data_
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
        model_name = "bert-base-multilingual-cased"
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

        for lang_, premise_, hypothesis_, labels_ in tqdm(
                zip(self.languages, self.premises, self.hypotheses, self.labels),
                leave=False, total=len(self.premises),
                desc=f'Indexing dataset to "{self.device}" using BERT Tokenizer'):
            encoded_dict = bert_tokenizer.encode_plus(text=premise_,
                                                      text_pair=hypothesis_,
                                                      add_special_tokens=True,
                                                      return_token_type_ids=True,
                                                      return_attention_mask=False,
                                                      return_tensors='pt')
            self.encoded_data.append({'languages': lang_,
                                      'premises_hypotheses': torch.squeeze(encoded_dict["input_ids"]),
                                      'token_types': torch.squeeze(encoded_dict["token_type_ids"]),
                                      'outputs': torch.LongTensor([self.label2idx.get(labels_)])})

    @staticmethod
    def pad_batch(batch):
        """ Pads sequences per batch with `padding_value=0`
        Args: batch: List[dict]
        Returns: dict of models inputs padded as per max len in batch
        """
        # ('languages', 'premises_hypotheses', 'token_types','outputs')
        languages_batch = [sample["languages"] for sample in batch]
        premises_hypotheses_batch = [sample["premises_hypotheses"] for sample in batch]
        padded_premises_hypotheses = pad_sequence(premises_hypotheses_batch, batch_first=True)
        # Token types is padded with 1 (unlike others padded with 0), padded with 1's due to
        # that we are doing pair_sentences encoding so the other sentence is consisting of 1's
        token_types_batch = [sample["token_types"] for sample in batch]
        padded_token_types = pad_sequence(token_types_batch, padding_value=1, batch_first=True)
        outputs_batch = pad_sequence([sample["outputs"] for sample in batch], batch_first=True)

        return {"languages": languages_batch,
                "premises_hypotheses": padded_premises_hypotheses,
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


class XLMDatasetParser(Dataset):
    """
    XLMDatasetParser builds the label2idx and idx2label, uses XLM Tokenizer, lowers all tokens
    indexes the dataset and pads per batch to feed model sequences of the same length
    """

    def __init__(self, _device, data_, model_="xlm-mlm-tlm-xnli15-1024"):
        super(XLMDatasetParser).__init__()
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
        """ Indexing and encoding the dataset using XLMTokenizer whilst lowering all tokens
        Tokenizer takes the sentence A (premises) and sentence B (hypotheses) adds to them the special tokens
        [CLS] & [SEP], it pads sentences to max length of 128 and truncates the ones with lengthier number of tokens
            - Original Sequence (inputs_ids) is changed to its corresponding IDs
            - Attention masks helps the model differntiate between [PAD] and original tokens
            - Token types helps the model know which sentence is Sentence A and which is the sentence B
        """
        tokenizer = XLMTokenizer.from_pretrained(self.model_name, do_lower_case=True)

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

            self.encoded_data.append({
                'languages': torch.LongTensor(
                    [tokenizer.lang2id.get(lang_) for _ in range(len(encoded_dict["input_ids"][0]))]),
                'premises_hypotheses': torch.squeeze(encoded_dict["input_ids"]),
                'attention_mask': torch.squeeze(encoded_dict["attention_mask"]),
                'token_types': torch.squeeze(encoded_dict["token_type_ids"]),
                'outputs': torch.LongTensor([self.label2idx.get(labels_)])
            })

    @staticmethod
    def pad_batch(batch):
        """ Pads sequences per batch
        Args: batch: List[dict]
        Returns: dict of models inputs padded as per max len in batch
        """
        # ('languages', 'premises_hypotheses', 'token_types','outputs')
        premises_hypotheses_batch = [sample["premises_hypotheses"] for sample in batch]
        # XLM Tokenizer padding -special token- index is 2, hence padding with value 2
        padded_premises_hypotheses = pad_sequence(premises_hypotheses_batch, padding_value=2, batch_first=True)

        # Padding the language sequences, in order for every word to be labeled with its language
        # for the model to know which sequence belongs to which language
        languages_batch = [sample["languages"] for sample in batch]
        lang_padding_val = int(languages_batch[0][0])
        padded_languages_batch = pad_sequence(languages_batch, padding_value=lang_padding_val, batch_first=True)

        mask = [sample["attention_mask"] for sample in batch]
        padded_mask = pad_sequence(mask, padding_value=0, batch_first=True)

        # Token types is padded with 1 (unlike others padded with 0), padded with 1's due to
        # that we are doing pair_sentences encoding so the other sentence is consisting of 1's
        token_types_batch = [sample["token_types"] for sample in batch]
        padded_token_types = pad_sequence(token_types_batch, padding_value=1, batch_first=True)
        outputs_batch = pad_sequence([sample["outputs"] for sample in batch], batch_first=True)

        return {"languages": padded_languages_batch,
                "premises_hypotheses": padded_premises_hypotheses,
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


class XLMRDatasetParser(Dataset):
    """
    XLMRDatasetParser identical to XLMRDatasetParser but without passing to the model the languages sequences
    it builds the label2idx and idx2label, uses XLM Tokenizer, lowers all tokens,
    indexes the dataset and pads per batch to feed model sequences of the same length
    """

    def __init__(self, _device, data_, model_="bert-base-multilingual-uncased"):
        super(XLMRDatasetParser).__init__()
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
        """ Indexing and encoding the dataset using XLMTokenizer whilst lowering all tokens
        Tokenizer takes the sentence A (premises) and sentence B (hypotheses) adds to them the special tokens
        [CLS] & [SEP], it pads sentences to max length of 128 and truncates the ones with lengthier number of tokens
            - Original Sequence (inputs_ids) is changed to its corresponding IDs
            - Attention masks helps the model differntiate between [PAD] and original tokens
            - Token types helps the model know which sentence is Sentence A and which is the sentence B
        """

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
        # XLM-R tokenizer padding index is 2, hence it is the padding value in `padded_premises_hypotheses`
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


if __name__ == "__main__":
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "word_stoi.pkl")
    nltk.download('punkt', quiet=True)
    model_name_ = "xlm-mlm-tlm-xnli15-1024"

    data = read_train()
    train_dataset = XLMDatasetParser(device_, data, model_name_)
    train_dataset.encode_dataset()

    dev_data = read_dev()
    dev_dataset = XLMDatasetParser(device_, dev_data, model_name_)
    dev_dataset.encode_dataset()

    test_data = read_test()
    test_dataset = XLMDatasetParser(device_, test_data, model_name_)
    test_dataset.encode_dataset()

    counter = collections.Counter(train_dataset.labels)
    print(f"Number of classes occurrences (train dataset):\n{counter}\n")
    train_df = pd.DataFrame(list(zip(train_dataset.languages,
                                     train_dataset.premises,
                                     train_dataset.hypotheses,
                                     train_dataset.labels)),
                            columns=['Languages', 'Premises',
                                     'Hypotheses', 'Labels'])

    print(train_df.head())

    dev_counter = collections.Counter(dev_dataset.labels)
    print(f"Number of classes occurrences (dev dataset):\n{dev_counter}\n")

    dev_df = pd.DataFrame(list(zip(dev_dataset.languages,
                                   dev_dataset.premises,
                                   dev_dataset.hypotheses,
                                   dev_dataset.labels)),
                          columns=['Languages', 'Premises',
                                   'Hypotheses', 'Labels'])

    print(dev_df.head())

    test_counter = collections.Counter(test_dataset.labels)
    print(f"Number of classes occurrences (test dataset):\n{test_counter}\n")

    test_df = pd.DataFrame(list(zip(test_dataset.languages,
                                    test_dataset.premises,
                                    test_dataset.hypotheses,
                                    test_dataset.labels)),
                           columns=['Languages', 'Premises',
                                    'Hypotheses', 'Labels'])

    print(test_df.head())

    Accuracy = pd.DataFrame()
    Accuracy['Type'] = train_df.Labels.value_counts().index
    Accuracy['Count'] = train_df.Labels.value_counts().values
    Accuracy['Type'] = Accuracy['Type'].replace(0, 'Entailment')
    Accuracy['Type'] = Accuracy['Type'].replace(1, 'Neutral')
    Accuracy['Type'] = Accuracy['Type'].replace(2, 'Contradiction')

    print(Accuracy)

    fig = go.Figure(data=[go.Pie(labels=Accuracy['Type'], values=Accuracy['Count'])])
    fig.update_layout(title={
        'text': "Percentage distribution of the 3 classes",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()

    fig = px.bar(Accuracy, x='Type', y='Count',
                 hover_data=['Count'], color='Count',
                 labels={'pop': 'Total Number of game titles'})

    fig.update_layout(title={
        'text': "Count of each of the target classes",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Count of each of the target classes (TRAIN).png")

    Languages = pd.DataFrame()
    Languages['Type'] = train_df.Languages.value_counts().index
    Languages['Count'] = train_df.Languages.value_counts().values

    fig = go.Figure(
        data=[go.Pie(labels=Languages['Type'], values=Languages['Count'])])
    fig.update_layout(title={
        'text': "Percentage distribution of different Languages",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Percentage distribution of different Languages (TEST).png")
    # From the above graph, we can see that no language is the dominating language in the test dataset.

    Test_Languages = pd.DataFrame()
    Test_Languages['Type'] = test_df.Languages.value_counts().index
    Test_Languages['Count'] = test_df.Languages.value_counts().values

    a = sum(Test_Languages.Count)
    Test_Languages.Count = Test_Languages.Count.div(a).mul(100).round(2)
    a = sum(Languages.Count)
    Languages.Count = Languages.Count.div(a).mul(100).round(2)

    fig = go.Figure(data=[
        go.Bar(name='Train', x=Languages.Type, y=Languages.Count),
        go.Bar(name='Test', x=Test_Languages.Type, y=Test_Languages.Count)
    ])
    # Change the bar mode
    fig.update_layout(barmode='group', title={
        'text': "Distribution across Train and Test",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Languages distribution across Train and Test.png")


    # From the graph, we can clearly see that the distribution of languages across train and test data are equal.

    def print_summary(summary_data, verbose=False):
        premises, dev_premises, test_premises, word2idx, label2idx = summary_data
        if verbose:
            print("\n=============Data Summary======================",
                  f"train_x length: {len(premises)} sentences",
                  f"dev_x length: {len(dev_premises)} sentences",
                  f"test_x length: {len(test_premises)} sentences",
                  f"Vocab size: {len(word2idx)}",
                  f"Labels vocab size: {len(label2idx)}",
                  "===============================================\n", sep="\n")


    """#### Meta features between Premesis and Hypothesis."""

    Meta_features = pd.DataFrame()

    ## Number of words in the text ##
    Meta_features["premise_num_words"] = train_df["Premises"].apply(
        lambda x: len(str(x).split()))
    Meta_features["hypothesis_num_words"] = train_df["Hypotheses"].apply(
        lambda x: len(str(x).split()))

    ## Average length of the words in the text ##
    Meta_features["premise_mean_word_len"] = train_df["Premises"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))
    Meta_features["hypothesis_mean_word_len"] = train_df["Hypotheses"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))

    Meta_features['Labels'] = train_df['Labels']


    def word_count(dataset, column):
        len_vector = []
        for text in dataset[column]:
            len_vector.append(len(text.split()))

        return len_vector


    train_premise = word_count(train_df, 'Premises')
    train_hypothesis = word_count(train_df, 'Hypotheses')

    dev_premise = word_count(dev_df, 'Premises')
    dev_hypothesis = word_count(dev_df, 'Hypotheses')

    test_premise = word_count(test_df, 'Premises')
    test_hypothesis = word_count(test_df, 'Hypotheses')

    fig = plt.figure(figsize=(25, 20))

    plt.subplot(3, 2, 1)
    plt.title('word count for train dataset premises')
    sns.distplot(train_premise)

    plt.subplot(3, 2, 2)
    plt.title('word count for train dataset hypotheses')
    sns.distplot(train_hypothesis)

    plt.subplot(3, 2, 3)
    plt.title('word count for dev dataset premises')
    sns.distplot(dev_premise)

    plt.subplot(3, 2, 4)
    plt.title('word count for dev dataset premises')
    sns.distplot(dev_premise)

    plt.subplot(3, 2, 5)
    plt.title('word count for test dataset premises')
    sns.distplot(test_premise)

    plt.subplot(3, 2, 6)
    plt.title('word count for test dataset hypotheses')
    sns.distplot(test_hypothesis)

    plt.savefig("word_counts.png")

    fig = plt.figure(figsize=(25, 20))

    plt.subplot(3, 2, 1)
    plt.title('word count for train dataset premise')
    sns.boxplot(train_premise)

    plt.subplot(3, 2, 2)
    plt.title('word count for train dataset hypothesis')
    sns.boxplot(train_hypothesis)

    plt.subplot(3, 2, 3)
    plt.title('word count for dev dataset premises')
    sns.boxplot(dev_premise)

    plt.subplot(3, 2, 4)
    plt.title('word count for dev dataset premises')
    sns.boxplot(dev_premise)

    plt.subplot(3, 2, 5)
    plt.title('word count for test dataset premise')
    sns.boxplot(test_premise)

    plt.subplot(3, 2, 6)
    plt.title('word count for test dataset hypothesis')
    sns.boxplot(test_hypothesis)

    plt.savefig("word_counts_boxplots.png")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=Meta_features['premise_num_words']))
    fig.add_trace(go.Histogram(x=Meta_features['hypothesis_num_words']))

    # Overlay both histograms
    fig.update_layout(barmode='overlay', title={
        'text': "Distribution of Words over Premise VS Hypothesis",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    # fig.show()
    fig.write_image("Distribution of Words over Premise VS Hypothesis.png")

    # plot wordcloud for premise (training)
    text = " ".join(txt for txt in train_df.Premises)
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(stopwords=stopwords, max_font_size=50,
                          max_words=500).generate(text)
    plt.figure(figsize=(25, 20))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    plt.savefig("english_word_cloud.png")

    STOPWORDS = set(stopwords.words('english'))
    PUNCTUATIONS = string.punctuation
    MLP_CNAMES = mlp.colors.cnames


    def random_color_generator(color_type=None):
        if color_type is None:
            colors = sorted(MLP_CNAMES.items(), key=lambda x: random())
        else:
            colors = sorted(color_type.items(), key=lambda x: random())
        return dict(colors)


    colors = random_color_generator()


    def show_word_cloud(data, title=None):
        word_cloud = WordCloud(
            background_color=list(colors.keys())[1],
            max_words=100,
            stopwords=STOPWORDS,
            max_font_size=40,
            scale=3,
            random_state=42).generate(data)
        fig = plt.figure(figsize=(20, 20))
        plt.axis('off')
        if title:
            fig.suptitle(title, fontsize=20)
            fig.subplots_adjust(top=2.3)

        plt.imshow(word_cloud)
        plt.show()
        fig.savefig(f"{str(title).replace(' ', '_').lower()}.png")


    # Most Comman words in entailment Prases
    entailment = " ".join(train_df[train_df.Labels == 'entailment']['Premises'])
    show_word_cloud(entailment, 'TOP 100 Entailment Words')

    # Most Comman words in Neutral Prases
    neutral = " ".join(train_df[train_df.Labels == 'neutral']['Premises'])
    show_word_cloud(neutral, 'TOP 100 Neutral Words')

    # Most Comman words in Contradictory Prases
    contradiction = " ".join(train_df[train_df.Labels == 'contradiction']['Premises'])
    show_word_cloud(contradiction, 'TOP 100 Contradiction Words')
