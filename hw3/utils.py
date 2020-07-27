from typing import Tuple, List, Any

import nlp


def flat_list(l: List[List[Any]]) -> List[Any]:
    return [_e for e in l for _e in e]


int2nli_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
nli_label2int = {v: k for k, v in int2nli_label.items()}
nli_labels = list(nli_label2int.keys())


def read_train() -> Tuple[List[str], List[str], List[str], List[str]]:
    return read_mnli(nlp.load_dataset('multi_nli')['train'])


def read_dev() -> Tuple[List[str], List[str], List[str], List[str]]:
    return read_mnli(nlp.load_dataset('multi_nli')['validation_matched'])


def read_test() -> Tuple[List[str], List[str], List[str], List[str]]:
    return read_xnli(nlp.load_dataset('xnli')['test'])


def read_mnli(dataset) -> Tuple[List[str], List[str], List[str], List[str]]:
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


def read_xnli(dataset) -> Tuple[List[str], List[str], List[str], List[str]]:
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


if __name__ == "__main__":
    languages, premises, hypotheses, labels = read_train()