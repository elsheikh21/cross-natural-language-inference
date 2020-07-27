import random
from typing import List

from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return your StudentModel
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return RandomBaseline()


class RandomBaseline(Model):
    """
    A very simple baseline to test that the evaluation script works
    """

    def __init__(self):
        self._labels = ['entailment', 'neutral', 'contradiction']

    def predict(self, languages: List[str], premises: List[str], hypotheses: List[str]) -> List[str]:
        return [random.choice(self._labels) for _, _ in zip(premises, hypotheses)]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

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
        pass
