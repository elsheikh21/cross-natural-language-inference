import os
from pathlib import Path

import nlp
import nltk
import torch
from nltk.tokenize import RegexpTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, XLMTokenizer
from utils import load_pickle, save_pickle
import pkbar


int2nli_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
nli_label2int = {v: k for k, v in int2nli_label.items()}
nli_labels = list(nli_label2int.keys())


def read_test():
    """
    Read test dataset (XNLI)
    """
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
    """
    Read training dataset (MNLI)
    """
    return read_mnli(nlp.load_dataset('multi_nli')['train'])


def read_dev():
    """
    Read validation dataset (MNLI)
    """
    return read_mnli(nlp.load_dataset('multi_nli')['validation_matched'])


class NLPDatasetParser(Dataset):
    def __init__(self, _device, data, is_bert):
        """
        Takes NLP Dataset (from Hugging Face NLP package),
        creates vocabulary, index dataset, pad per batch

        Args:
            data [List[str], List[str], List[str], List[str]]: A tuple of 4 lists containing
            language, premise, hypothesis, and labels
        """
        super(NLPDatasetParser).__init__()
        self.encoded_data = []
        self.device = _device
        self.languages, self.premises, self.hypotheses, self.labels = data
        self.use_bert = is_bert

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
        model_name = "bert-base-multilingual-cased"
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        lang_x, premises_x_stoi, hypotheses_x_stoi, data_y_stoi = [], [], [], []
        for lang_, premise_, hypothesis_, labels in tqdm(zip(self.languages, self.premises, self.hypotheses, self.labels),
                                                         leave=False, total=len(self.premises),
                                                         desc=f'Indexing dataset to {self.device}'):
            lang_x.append(lang_)
            if self.use_bert:
                input_ids = bert_tokenizer.encode(premise_, hypothesis_, return_tensors='pt')
                premises_x_stoi.append(torch.squeeze(input_ids))
            else:
                premises_x_stoi.append(torch.LongTensor([word2idx.get(word, 1) for word in premise_]))
                hypotheses_x_stoi.append(torch.LongTensor([word2idx.get(word, 1) for word in hypothesis_]))
            data_y_stoi.append(torch.LongTensor([label2idx.get(labels)]))

        if not self.use_bert:
            for i in tqdm(range(len(premises_x_stoi)), desc="Encoding dataset",
                          leave=False, total=len(self.premises)):
                self.encoded_data.append({'languages': lang_x[i],
                                          'premises': premises_x_stoi[i],
                                          'hypotheses': hypotheses_x_stoi[i],
                                          'outputs': data_y_stoi[i]})
        else:
            for i in tqdm(range(len(premises_x_stoi)), desc="Encoding dataset using BART tokenizer",
                          leave=False, total=len(self.premises)):
                self.encoded_data.append({'languages': lang_x[i],
                                          'premises_hypotheses': premises_x_stoi[i],
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
        premises_hypotheses_flag = True
        try:
            x = [sample["premises_hypotheses"] for sample in batch]
        except KeyError:
            premises_hypotheses_flag = False


        if not premises_hypotheses_flag:
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
        else:
            languages_batch = [sample["languages"] for sample in batch]
            premises_hypotheses_batch = [sample["premises_hypotheses"] for sample in batch]
            padded_premises_hypotheses = pad_sequence(premises_hypotheses_batch, batch_first=True)
            outputs_batch = pad_sequence([sample["outputs"] for sample in batch], batch_first=True)
            # outputs_batch = [sample["outputs"] for sample in batch]

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
    def __init__(self, _device, data, is_bert):
        """
        Takes NLP Dataset (from Hugging Face NLP package),
        creates vocabulary, index dataset, pad per batch

        Args:
            data [List[str], List[str], List[str], List[str]]: A tuple of 4 lists containing
            language, premise, hypothesis, and labels
        """
        super(K_NLPDatasetParser).__init__()
        self.encoded_data, self.premises_hypotheses = [], []
        self.languages, self.premises, self.hypotheses, self.labels = data
        self.device = _device
        self.use_bert = is_bert
        # if not self.use_bert:
        self.process_dataset()

    def create_vocabulary(self, load_from=None):
        """
        Creates vocab dictionaries for both data_x which is composed of 2 lists (premises and hypotheses), and
        creates labels vocab dictionary by invoking `create_labels_vocabulary()`.
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
        Args: word2idx (dict): Vocab dictionary & label2idx (dict): Labels dictionary
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
        """ Pads sequences per batch with `padding_value=0`
        Args: batch: List[dict]
        Returns: dict of models inputs padded as per max len in batch
        """
        languages_batch = [sample["languages"] for sample in batch]
        premises_hypotheses_batch = [sample["premises_hypotheses"] for sample in batch]
        padded_premises_hypotheses = pad_sequence(premises_hypotheses_batch, batch_first=True)
        outputs_batch = pad_sequence([sample["outputs"] for sample in batch], batch_first=True)
        # outputs_batch = [sample["outputs"] for sample in batch]

        return {'languages': languages_batch,
                'premises_hypotheses': padded_premises_hypotheses,
                'outputs': outputs_batch}

    def process_dataset(self):
        for premises_, hypotheses_ in tqdm(zip(self.premises, self.hypotheses), leave=False,
                                           desc='Processing Dataset', total=len(self.premises)):
            self.premises_hypotheses.append(f"{premises_.strip()} [SEP] {hypotheses_.strip()}")
        # lengthiest_seq = max(self.premises_hypotheses, key=len)
        # print(f"MAX LENGTH OF SEQ: {len(lengthiest_seq)} --- {lengthiest_seq}")
        del self.premises, self.hypotheses


class BERTDatasetParser(Dataset):
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
    """ PreTrainedModel (XLM) DatasetParser """

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
        """ Indexing and encoding the dataset """
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
                                                 max_length = 128,
                                                 truncation=True,
                                                 pad_to_max_length=True,
                                                 return_tensors='pt')
            self.encoded_data.append({'languages': torch.LongTensor([tokenizer.lang2id.get(lang_) for _ in range(len(encoded_dict["input_ids"][0]))]),
                                      'premises_hypotheses': torch.squeeze(encoded_dict["input_ids"]),
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


class XLMRTrainer:
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
                    state[k] = v.to(self.device)


if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    data = read_mnli(nlp.load_dataset('multi_nli')['train'])
    languages, premises, hypotheses, labels = data
    dev_lang, dev_premises, dev_hypotheses, dev_labels = read_mnli(nlp.load_dataset('multi_nli')['validation_matched'])

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "word_stoi.pkl")

    train_dataset = BERTDatasetParser(device_, data)
    train_dataset.encode_dataset()

    # train_dataset = NLPDatasetParser(device_, data)
    # word2idx, idx2word, label2idx, idx2label = train_dataset.create_vocabulary(save_path)
    #
    # print(f"Train data_x length: {len(premises)} sentences")
    # print(f"Dev data_x length: {len(dev_premises)} sentences")
    # print(f"Vocab size: {len(word2idx)}")
    # print(f"Labels vocab size: {len(label2idx)}")
    #
    # train_dataset.encode_dataset(word2idx, label2idx)
