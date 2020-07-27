import nlp

from tabulate import tabulate


def compute_metrics(languages, predicted_labels, labels):
    """
    Computes accuracy of the predicted labels (y_hat) vs labels (y)
    """
    metric = nlp.load_metric('xnli')
    headers = ('', 'accuracy', '# samples')
    table = []
    table.append(('overall', metric.compute(predicted_labels, labels)['accuracy'], len(labels)))

    # per language]
    for evaluation_language in sorted(set(languages)):
        evaluation_language_predicted_labels = [predicted_label for predicted_label, language in
                                                zip(predicted_labels, languages) if language == evaluation_language]
        evaluation_language_labels = [label for label, language in zip(labels, languages) if
                                      language == evaluation_language]
        evaluation_language_accuracy = metric.compute(evaluation_language_predicted_labels, evaluation_language_labels)[
            'accuracy']
        table.append((evaluation_language, evaluation_language_accuracy, len(evaluation_language_labels)))
    print(tabulate(table, headers=headers, tablefmt="pretty"))
