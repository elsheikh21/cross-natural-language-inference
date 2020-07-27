import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import argparse
import nlp
import requests
import time

from requests.exceptions import ConnectionError
from tabulate import tabulate
from tqdm import tqdm

from utils import nli_label2int, read_dev, read_test


_option2dataset = {
    'dev': read_dev(),
    'test': read_test(),
}


def main(dataset_option: str, endpoint: str, batch_size=32):

    try:
        languages, premises, hypotheses, labels = _option2dataset[dataset_option]
    except KeyError as e:
        logging.error(f'{e} is not a valid choice as a dataset_option')
        logging.error(f'This should have failed through argparse. Contact the teaching assistants.')
        exit(1)
    except Exception as e:
        logging.error(f'Evaluation crashed. Most likely, something happened with the nlp library')
        logging.error(f'Read the reported error and contact the teaching assistants')
        logging.error(e, exc_info=True)
        exit(1)

    max_try = 10
    iterator = iter(range(max_try))

    while True:

        try:
            i = next(iterator)
        except StopIteration:
            logging.error(f'Impossible to establish a connection to the server even after 10 tries')
            logging.error('The server is not booting and, most likely, you have some error in build_model or StudentClass')
            logging.error('You can find more information inside logs/. Checkout both server.stdout and, most importantly, server.stderr')
            exit(1)

        logging.info(f'Waiting 10 second for server to go up: trial {i}/{max_try}')
        time.sleep(10)

        try:
            response = requests.post(
                endpoint,
                json={
                    'languages': ['en'],
                    'premises': ['He shot him with a gun'],
                    'hypotheses': ['He has never used a gun']
                }
            ).json()
            response['predicted_labels']
            logging.info('Connection succeded')
            break
        except ConnectionError as e:
            continue
        except KeyError as e:
            logging.error(f'Server response in wrong format')
            logging.error(f'Response was: {response}')
            logging.error(e, exc_info=True)
            exit(1)

    predicted_labels = []
    progress_bar = tqdm(total=len(premises), desc='Evaluating')

    for i in range(0, len(premises), batch_size):

        batch_languages = languages[i: i + batch_size]
        batch_premises = premises[i: i + batch_size]
        batch_hypotheses = hypotheses[i: i + batch_size]
        assert len(batch_premises) == len(batch_hypotheses)

        try:
            response = requests.post(endpoint, json={'languages': batch_languages, 'premises': batch_premises, 'hypotheses': batch_hypotheses}).json()
            predicted_labels += response['predicted_labels']
        except KeyError as e:
            logging.error(f'Server response in wrong format')
            logging.error(f'Response was: {response}')
            logging.error(e, exc_info=True)
            exit(1)

        progress_bar.update(len(batch_premises))

    progress_bar.close()

    labels = [nli_label2int[label] for label in labels]
    predicted_labels = [nli_label2int[predicted_label] for predicted_label in predicted_labels]

    # load metric and prepare print
    metric = nlp.load_metric('xnli')
    headers = ('', 'accuracy', '# samples')
    table = []

    # overall
    table.append(('overall', metric.compute(predicted_labels, labels)['accuracy'], len(labels)))

    # per language
    for evaluation_language in sorted(set(languages)):
        evaluation_language_predicted_labels = [predicted_label for predicted_label, language in zip(predicted_labels, languages) if language == evaluation_language]
        evaluation_language_labels = [label for label, language in zip(labels, languages) if language == evaluation_language]
        evaluation_language_accuracy = metric.compute(evaluation_language_predicted_labels, evaluation_language_labels)['accuracy']
        table.append((evaluation_language, evaluation_language_accuracy, len(evaluation_language_labels)))

    # print
    print(tabulate(table, headers=headers, tablefmt="pretty"))


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_option", type=str, help='Dataset to test upon', choices=list(_option2dataset.keys()))
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    main(
        dataset_option=args.dataset_option,
        endpoint='http://127.0.0.1:12345'
    )
