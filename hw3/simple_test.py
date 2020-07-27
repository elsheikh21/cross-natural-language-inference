from typing import List

from nlp import DownloadConfig

from stud.implementation import build_model


def main(languages: List[str], premises: List[str], hypotheses: List[str]):

    model = build_model('cpu')
    predicted_labels = model.predict(languages, premises, hypotheses)

    for language, premise, hypothesis, predicted_label in zip(languages, premises, hypotheses, predicted_labels):
        print(f'# language = {language}')
        print(f'# premise = {premise}')
        print(f'# hypothesis = {hypothesis}')
        print(f'# label = {predicted_label}')
        print()


if __name__ == '__main__':
    main(['en'], ['He shot him with a gun'], ['He has never used a gun'])
