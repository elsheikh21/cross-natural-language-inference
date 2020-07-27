from typing import List


class Model:

    # STUDENT: DO NOT touch this file
    # STUDENT: it is simply the abstract class behind your models

    def predict(self, languages: List[str], premises: List[str], hypotheses: List[str]) -> List[str]:
        """

        Args:
            languages (list): list of languages
            premises (list): list of premises
            hypotheses (list): list of hypotheses

        Returns:
            list: predicted labels

        """
        raise NotImplementedError
