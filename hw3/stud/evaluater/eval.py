import nlp

from tabulate import tabulate


def compute_metrics(languages, predicted_labels, labels):
    """
    step 1. Convert tags of both labels and predicted_labels to indices
    step 2. Computes accuracy of the predicted labels (y_hat) vs labels (y)
    """
    int2nli_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    nli_label2int = {v: k for k, v in int2nli_label.items()}

    labels = [nli_label2int[label] for label in labels]
    predicted_labels = [nli_label2int[predicted_label] for predicted_label in predicted_labels]

    metric = nlp.load_metric('xnli')
    headers = ('', 'accuracy', '# samples')
    table = []
    table.append(('overall', metric.compute(predicted_labels, labels)['accuracy'], len(labels)))

    # per language
    for evaluation_language in sorted(set(languages)):
        evaluation_language_predicted_labels = [predicted_label for predicted_label, language in
                                                zip(predicted_labels, languages) if language == evaluation_language]
        evaluation_language_labels = [label for label, language in zip(labels, languages) if
                                      language == evaluation_language]
        evaluation_language_accuracy = metric.compute(evaluation_language_predicted_labels, evaluation_language_labels)[
            'accuracy']
        table.append((evaluation_language, evaluation_language_accuracy, len(evaluation_language_labels)))
    print(tabulate(table, headers=headers, tablefmt="pretty"))
