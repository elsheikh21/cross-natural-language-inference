import random
from typing import List
import os
import torch
from model import Model
from transformers import AutoTokenizer
from stud.models import HyperParameters, XLMR_Model


def build_model(device: str) -> Model:
    # STUDENT: return your StudentModel
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)


class RandomBaseline(Model):
    """
    A very simple baseline to test that the evaluation script works
    """
    def __init__(self):
        self._labels = ['entailment', 'neutral', 'contradiction']

    def predict(self, languages: List[str], premises: List[str], hypotheses: List[str]) -> List[str]:
        return [random.choice(self._labels) for _, _ in zip(premises, hypotheses)]


class StudentModel(Model):
    def __init__(self, device_):
        self.device = device_
        self.label2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.idx2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        model_name = "xlm-roberta-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

        self.hp = HyperParameters(model_name_='XLM-R_Seq_CLS', vocab=None,
                                  label_vocab=self.label2idx, embeddings_=None, batch_size_=8)                                                                                                                                          
        self.model = XLMR_Model(hparams=self.hp, model_name=model_name, freeze_model=False).to(self.device)

    def predict(self, languages: List[str], premises: List[str], hypotheses: List[str]) -> List[str]:
        """
        STUDENT: implement here your predict function
        Args:
            languages (list): list of languages
            premises (list): list of premises
            hypotheses (list): list of hypotheses

        Returns:
            list: predicted labels
        """
        # Load the model
        model_path = os.path.join(os.getcwd(), 'model', 'XLM-R_Seq_CLS_ckpt_best.pth')
        self.model.load_(path=model_path)

        # Encode the dataset passed to the model
        encoded_dict = self.tokenizer.encode_plus(text=premises, text_pair=hypotheses,
                                                  add_special_tokens=True, return_token_type_ids=True,
                                                  return_attention_mask=True, truncation=True, max_length=128,
                                                  pad_to_max_length=True, return_tensors='pt')
        seq = torch.LongTensor(encoded_dict["input_ids"]).to(self.device)
        mask = torch.LongTensor(encoded_dict["attention_mask"]).to(self.device)
        tokens_type = torch.LongTensor(encoded_dict["token_type_ids"]).to(self.device)

        # Predict data from model
        predicted_labels = self.model.predict_sentence_(seq, mask, tokens_type)
        return [self.idx2label.get(tag) for tag in predicted_labels]
        # FIXME: pyarrow.lib.ArrowInvalid: Column 1 named references expected length 2349 but got length 75150