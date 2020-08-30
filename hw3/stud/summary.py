import collections
import random
import string
from pprint import pprint
from random import random

import matplotlib as mlp
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from hw3.stud.data_loader import XLMRDatasetParser, read_train, read_dev, read_test
import torch
import os


def word_count(dataset, column):
    len_vector = []
    for text in dataset[column]:
        len_vector.append(len(text.split()))
    return len_vector


def random_color_generator(color_type=None):
    MLP_CNAMES = mlp.colors.cnames
    if color_type is None:
        colors = sorted(MLP_CNAMES.items(), key=lambda x: random())
    else:
        colors = sorted(color_type.items(), key=lambda x: random())
    return dict(colors)


def show_word_cloud(data, title=None):
    colors = random_color_generator()
    STOPWORDS = set(stopwords.words('english'))
    word_cloud = WordCloud(
        background_color=list(colors.keys())[1],
        max_words=100,
        stopwords=STOPWORDS,
        max_font_size=40,
        scale=3,
        random_state=42).generate(data)
    fig = plt.figure(figsize=(20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(word_cloud)
    plt.show()
    fig.savefig(f"{str(title).replace(' ', '_').lower()}.png")


def print_summary(summary_data, verbose=False):
    premises, dev_premises, test_premises, word2idx, label2idx = summary_data
    # To clear out cell from unwanted downloading and indexing progress bars
    if verbose:
        print("\n=============Data Summary======================",
              f"train_x length: {len(premises)} sentences",
              f"dev_x length: {len(dev_premises)} sentences",
              f"test_x length: {len(test_premises)} sentences",
              f"Vocab size: {len(word2idx)}",
              f"Labels vocab size: {len(label2idx)}",
              "===============================================\n", sep="\n")


def labels_freq(dataset_, name):
    """ Computes labels distribution and returns a dataframe"""
    counter = collections.Counter(dataset_.labels)
    print(f"Number of classes occurrences ({name} dataset):\n{counter}\n")
    dataset_df = pd.DataFrame(list(zip(dataset_.languages, dataset_.premises, dataset_.hypotheses, dataset_.labels)),
                            columns=['Languages', 'Premises', 'Hypotheses', 'Labels'])
    return dataset_df


def get_labels_distribution_in_dataframe(data_dataframe_):
    Accuracy = pd.DataFrame()
    Accuracy['Type'] = data_dataframe_.Labels.value_counts().index
    Accuracy['Count'] = data_dataframe_.Labels.value_counts().values
    Accuracy['Type'] = Accuracy['Type'].replace(0, 'Entailment')
    Accuracy['Type'] = Accuracy['Type'].replace(1, 'Neutral')
    Accuracy['Type'] = Accuracy['Type'].replace(2, 'Contradiction')
    return Accuracy


def datasets_languages_distribution(train_df, test_df):
    """ fetches the percentage distribution of languages in provided dataframes and save the corresponding figures """
    languages = pd.DataFrame()
    languages['Type'] = train_df.Languages.value_counts().index
    languages['Count'] = train_df.Languages.value_counts().values

    fig = go.Figure(data=[go.Pie(labels=languages['Type'],
                                 values=languages['Count'])])
    fig.update_layout(title={'text': "Percentage distribution of different Languages",
                             'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
    fig.show()
    fig.write_image("Percentage distribution of different Languages (TEST).png")
    # From the above graph, we can see that no language is the dominating language in the test dataset.

    test_languages = pd.DataFrame()
    test_languages['Type'] = test_df.Languages.value_counts().index
    test_languages['Count'] = test_df.Languages.value_counts().values

    a = sum(test_languages.Count)
    test_languages.Count = test_languages.Count.div(a).mul(100).round(2)
    a = sum(languages.Count)
    languages.Count = languages.Count.div(a).mul(100).round(2)

    fig = go.Figure(data=[go.Bar(name='Train', x=languages.Type, y=languages.Count),
                          go.Bar(name='Test', x=test_languages.Type, y=test_languages.Count)
                          ]
                    )
    # Change the bar mode
    fig.update_layout(barmode='group', title={
        'text': "Distribution across Train and Test",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Languages distribution across Train and Test.png")
    # From the graph, we can clearly see that the distribution of languages across train and test data are equal.
    fig = go.Figure(
        data=[go.Pie(labels=languages['Type'], values=languages['Count'])])
    fig.update_layout(title={
        'text': "Percentage distribution of the 3 classes",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Percentage distribution of the 3 classes (TRAIN).png")
    # From the above graph, we can see that English is the dominating language in the train & dev datasets.

    fig = px.bar(test_languages, x='Type', y='Count',
                 hover_data=['Count'], color='Count',
                 labels={'pop': 'Total Number of game titles'})

    fig.update_layout(title={
        'text': "Count of each of the target classes",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Count of each of the target classes (TEST).png")
    # From the above graph, we can see that English is the dominating language in the train & sdev datasets.



def sentences_length_analysis(dataframes_):
    train_df, dev_df, test_df = dataframes_
    train_premise = word_count(train_df, 'Premises')
    train_hypothesis = word_count(train_df, 'Hypotheses')
    dev_premise = word_count(dev_df, 'Premises')
    dev_hypothesis = word_count(dev_df, 'Hypotheses')
    test_premise = word_count(test_df, 'Premises')
    test_hypothesis = word_count(test_df, 'Hypotheses')
    fig = plt.figure(figsize=(25, 20))
    plt.subplot(3, 2, 1)
    plt.title('word count for train dataset premises')
    sns.distplot(train_premise)
    plt.subplot(3, 2, 2)
    plt.title('word count for train dataset hypotheses')
    sns.distplot(train_hypothesis)
    plt.subplot(3, 2, 3)
    plt.title('word count for dev dataset premises')
    sns.distplot(dev_premise)
    plt.subplot(3, 2, 4)
    plt.title('word count for dev dataset premises')
    sns.distplot(dev_premise)
    plt.subplot(3, 2, 5)
    plt.title('word count for test dataset premises')
    sns.distplot(test_premise)
    plt.subplot(3, 2, 6)
    plt.title('word count for test dataset hypotheses')
    sns.distplot(test_hypothesis)
    plt.savefig("word_counts.png")

    fig = plt.figure(figsize=(25, 20))
    plt.subplot(3, 2, 1)
    plt.title('word count for train dataset premise')
    sns.boxplot(train_premise)
    plt.subplot(3, 2, 2)
    plt.title('word count for train dataset hypothesis')
    sns.boxplot(train_hypothesis)
    plt.subplot(3, 2, 3)
    plt.title('word count for dev dataset premises')
    sns.boxplot(dev_premise)
    plt.subplot(3, 2, 4)
    plt.title('word count for dev dataset premises')
    sns.boxplot(dev_premise)
    plt.subplot(3, 2, 5)
    plt.title('word count for test dataset premise')
    sns.boxplot(test_premise)
    plt.subplot(3, 2, 6)
    plt.title('word count for test dataset hypothesis')
    sns.boxplot(test_hypothesis)
    plt.savefig("word_counts_boxplots.png")


def premises_hypotheses_words_distribution(train_df):
    Meta_features = pd.DataFrame()
    ## Number of words in the text ##
    Meta_features["premise_num_words"] = train_df["Premises"].apply(
        lambda x: len(str(x).split()))
    Meta_features["hypothesis_num_words"] = train_df["Hypotheses"].apply(
        lambda x: len(str(x).split()))

    ## Average length of the words in the text ##
    Meta_features["premise_mean_word_len"] = train_df["Premises"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))
    Meta_features["hypothesis_mean_word_len"] = train_df["Hypotheses"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]))

    Meta_features['Labels'] = train_df['Labels']

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=Meta_features['premise_num_words']))
    fig.add_trace(go.Histogram(x=Meta_features['hypothesis_num_words']))

    # Overlay both histograms
    fig.update_layout(barmode='overlay', title={
        'text': "Distribution of Words over Premise VS Hypothesis",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.write_image("Distribution of Words over Premise VS Hypothesis.png")
    fig.show()


def plot_word_clouds(train_df):
    training_data = []
    training_data.extend(train_df.Premises)
    training_data.extend(train_df.hypotheses)
    text = " ".join(txt for txt in training_data)
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(stopwords=stopwords, max_font_size=50,
                          max_words=500).generate(text)
    plt.figure(figsize=(25, 20))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    plt.savefig("english_word_cloud.png")

    entailment = " ".join(train_df[train_df.Labels == 'entailment'])
    show_word_cloud(entailment, 'TOP 100 Entailment Words')

    neutral = " ".join(train_df[train_df.Labels == 'neutral'])
    show_word_cloud(neutral, 'TOP 100 Neutral Words')

    contradiction = " ".join(train_df[train_df.Labels == 'contradiction'])
    show_word_cloud(contradiction, 'TOP 100 Contradiction Words')


if __name__ == "__main__":
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "word_stoi.pkl")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    model_name_ = "xlm-mlm-tlm-xnli15-1024"

    data = read_train()
    train_dataset = XLMRDatasetParser(device_, data, model_name_)
    train_dataset.encode_dataset()

    dev_data = read_dev()
    dev_dataset = XLMRDatasetParser(device_, dev_data, model_name_)
    dev_dataset.encode_dataset()

    test_data = read_test()
    test_dataset = XLMRDatasetParser(device_, test_data, model_name_)
    test_dataset.encode_dataset()

    train_df = labels_freq(train_dataset, 'train')
    print(train_df.head())
    dev_df = labels_freq(dev_dataset, 'dev')
    print(dev_df.head())
    test_df = labels_freq(test_dataset, 'test')
    print(test_df.head())

    train_labels_dist = get_labels_distribution_in_dataframe(train_df)
    pprint(train_labels_dist)
    dev_labels_dist = get_labels_distribution_in_dataframe(dev_df)
    pprint(dev_labels_dist)
    test_labels_dist = get_labels_distribution_in_dataframe(test_df)
    pprint(test_labels_dist)

    datasets_languages_distribution(train_df, test_df)

    premises_hypotheses_words_distribution(train_df)

    dataframes = train_df, dev_df, test_df
    sentences_length_analysis(dataframes)

    plot_word_clouds(train_df)
