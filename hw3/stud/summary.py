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

    counter = collections.Counter(train_dataset.labels)
    print(f"Number of classes occurrences (train dataset):\n{counter}\n")
    train_df = pd.DataFrame(list(zip(train_dataset.languages,
                                     train_dataset.premises,
                                     train_dataset.hypotheses,
                                     train_dataset.labels)),
                            columns=['Languages', 'Premises',
                                     'Hypotheses', 'Labels'])

    pprint(train_df.head())

    dev_counter = collections.Counter(dev_dataset.labels)
    print(f"Number of classes occurrences (dev dataset):\n{dev_counter}\n")

    dev_df = pd.DataFrame(list(zip(dev_dataset.languages,
                                   dev_dataset.premises,
                                   dev_dataset.hypotheses,
                                   dev_dataset.labels)),
                          columns=['Languages', 'Premises',
                                   'Hypotheses', 'Labels'])

    pprint(dev_df.head())

    test_counter = collections.Counter(test_dataset.labels)
    print(f"Number of classes occurrences (test dataset):\n{test_counter}\n")

    test_df = pd.DataFrame(list(zip(test_dataset.languages,
                                    test_dataset.premises,
                                    test_dataset.hypotheses,
                                    test_dataset.labels)),
                           columns=['Languages', 'Premises',
                                    'Hypotheses', 'Labels'])

    pprint(test_df.head())

    Accuracy = pd.DataFrame()
    Accuracy['Type'] = train_df.Labels.value_counts().index
    Accuracy['Count'] = train_df.Labels.value_counts().values
    Accuracy['Type'] = Accuracy['Type'].replace(0, 'Entailment')
    Accuracy['Type'] = Accuracy['Type'].replace(1, 'Neutral')
    Accuracy['Type'] = Accuracy['Type'].replace(2, 'Contradiction')
    pprint(Accuracy)

    dev_Accuracy = pd.DataFrame()
    dev_Accuracy['Type'] = dev_df.Labels.value_counts().index
    dev_Accuracy['Count'] = dev_df.Labels.value_counts().values
    dev_Accuracy['Type'] = dev_Accuracy['Type'].replace(0, 'Entailment')
    dev_Accuracy['Type'] = dev_Accuracy['Type'].replace(1, 'Neutral')
    dev_Accuracy['Type'] = dev_Accuracy['Type'].replace(2, 'Contradiction')
    pprint(dev_Accuracy)

    test_Accuracy = pd.DataFrame()
    test_Accuracy['Type'] = test_df.Labels.value_counts().index
    test_Accuracy['Count'] = test_df.Labels.value_counts().values
    test_Accuracy['Type'] = test_Accuracy['Type'].replace(0, 'Entailment')
    test_Accuracy['Type'] = test_Accuracy['Type'].replace(1, 'Neutral')
    test_Accuracy['Type'] = test_Accuracy['Type'].replace(2, 'Contradiction')
    pprint(test_Accuracy)

    fig = go.Figure(
        data=[go.Pie(labels=Accuracy['Type'], values=Accuracy['Count'])])
    fig.update_layout(title={
        'text': "Percentage distribution of the 3 classes",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Percentage distribution of the 3 classes (TRAIN).png")
    # From the above graph, we can see that English is the dominating language in the train & dev datasets.

    fig = px.bar(dev_Accuracy, x='Type', y='Count',
                 hover_data=['Count'], color='Count',
                 labels={'pop': 'Total Number of game titles'})

    fig.update_layout(title={
        'text': "Count of each of the target classes",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Count of each of the target classes (DEV).png")
    # From the above graph, we can see that English is the dominating language in the train & sdev datasets.

    Languages = pd.DataFrame()
    Languages['Type'] = train_df.Languages.value_counts().index
    Languages['Count'] = train_df.Languages.value_counts().values

    fig = go.Figure(
        data=[go.Pie(labels=Languages['Type'], values=Languages['Count'])])
    fig.update_layout(title={
        'text': "Percentage distribution of different Languages",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.show()
    fig.write_image("Percentage distribution of different Languages (TEST).png")
    # From the above graph, we can see that no language is the dominating language in the test dataset.

    Test_Languages = pd.DataFrame()
    Test_Languages['Type'] = test_df.Languages.value_counts().index
    Test_Languages['Count'] = test_df.Languages.value_counts().values

    a = sum(Test_Languages.Count)
    Test_Languages.Count = Test_Languages.Count.div(a).mul(100).round(2)
    a = sum(Languages.Count)
    Languages.Count = Languages.Count.div(a).mul(100).round(2)

    fig = go.Figure(data=[
        go.Bar(name='Train', x=Languages.Type, y=Languages.Count),
        go.Bar(name='Test', x=Test_Languages.Type, y=Test_Languages.Count)
    ])
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


    """#### Meta features between Premesis and Hypothesis."""

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


    def word_count(dataset, column):
        len_vector = []
        for text in dataset[column]:
            len_vector.append(len(text.split()))

        return len_vector


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
    # fig.show()
    fig.write_image("Distribution of Words over Premise VS Hypothesis.png")

    # plot wordcloud for premise (training)
    text = " ".join(txt for txt in train_df.Premises)
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(stopwords=stopwords, max_font_size=50,
                          max_words=500).generate(text)
    plt.figure(figsize=(25, 20))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    plt.savefig("english_word_cloud.png")

    # Most Comman words in entailment Prases
    entailment = " ".join(train_df[train_df.Labels == 'entailment']['Premises'])
    show_word_cloud(entailment, 'TOP 100 Entailment Words')

    # Most Comman words in Neutral Prases
    neutral = " ".join(train_df[train_df.Labels == 'neutral']['Premises'])
    show_word_cloud(neutral, 'TOP 100 Neutral Words')

    # Most Comman words in Contradictory Prases
    contradiction = " ".join(train_df[train_df.Labels == 'contradiction']['Premises'])
    show_word_cloud(contradiction, 'TOP 100 Contradiction Words')
