import os
from pathlib import Path

import nlp
import nltk
import torch
from nltk.tokenize import RegexpTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils import load_pickle, save_pickle


int2nli_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
nli_label2int = {v: k for k, v in int2nli_label.items()}
nli_labels = list(nli_label2int.keys())


def read_test():
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
    return read_mnli(nlp.load_dataset('multi_nli')['train'])


class NLPDatasetParser(Dataset):
    def __init__(self, _device, data):
        """
        Takes NLP Dataset (from Hugging Face NLP package),
        creates vocabulary, index dataset, pad per batch

        Args:
            data [List[str], List[str], List[str], List[str]]: A tuple of 4 lists containing
            language, premise, hypothesis, and labels
        """
        super().__init__()
        self.encoded_data = []
        self.device = _device
        self.languages, self.premises, self.hypotheses, self.labels = data

    def create_vocabulary(self, load_from=None):
        """
        Creates vocab dictionaries for both data_x which is composed of 2 lists (premeses and hypotheses), and
        creates labels vocab dictionary by invoking `create_labels_vocabulary()`.
        Auxiliary functionalities: saves the vocab dict


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
        tokenizer = RegexpTokenizer("[\w']+")
        premises_words = [word for word in tokenizer.tokenize(" ".join(self.premises))]
        hypotheses_words = [word for word in tokenizer.tokenize(" ".join(self.hypotheses))]
        premises_words.extend(hypotheses_words)
        unigrams = sorted(list(set(premises_words)))
        stoi = {'<PAD>': 0, '<UNK>': 1}
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
        lang_x, premises_x_stoi, hypotheses_x_stoi, data_y_stoi = [], [], [], []
        for lang_, premise_, hypothesis_, labels in tqdm(
                zip(self.languages, self.premises, self.hypotheses, self.labels),
                leave=False, total=len(self.premises),
                desc=f'Indexing dataset to {self.device}'):
            lang_x.append(lang_)
            premises_x_stoi.append(torch.LongTensor([word2idx.get(word, 1) for word in premise_]))
            hypotheses_x_stoi.append(torch.LongTensor([word2idx.get(word, 1) for word in hypothesis_]))
            data_y_stoi.append(torch.LongTensor([label2idx.get(labels)]))

        for i in tqdm(range(len(premises_x_stoi)), desc="Encoding dataset",
                      leave=False, total=len(self.premises) ):
            self.encoded_data.append({'languages': lang_x[i],
                                      'premises': premises_x_stoi[i],
                                      'hypotheses': hypotheses_x_stoi[i],
                                      'outputs': data_y_stoi[i]})

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("Dataset is not indexed yet.\
                                To fetch raw elements, use get_element(idx)")
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
        languages_batch = [sample["languages"] for sample in batch]
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

        outputs_batch = pad_sequence([sample["outputs"] for sample in batch], batch_first=True)

        return {'languages': languages_batch,
                'premises': padded_premise,
                'hypotheses': padded_hypotheses,
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


if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    data = read_mnli(nlp.load_dataset('multi_nli')['train'])
    languages, premises, hypotheses, labels = data
    dev_lang, dev_premises, dev_hypotheses, dev_labels = read_mnli(nlp.load_dataset('multi_nli')['validation_matched'])

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "word_stoi.pkl")
    train_dataset = NLPDatasetParser(device_, data)
    word2idx, idx2word, label2idx, idx2label = train_dataset.create_vocabulary(save_path)

    print(f"Train data_x length: {len(premises)} sentences")
    print(f"Dev data_x length: {len(dev_premises)} sentences")
    print(f"Vocab size: {len(word2idx)}")
    print(f"Labels vocab size: {len(label2idx)}")

    train_dataset.encode_dataset(word2idx, label2idx)
