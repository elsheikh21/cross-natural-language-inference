import datetime
import glob
import io
import logging
import os
import pickle
import random

import nltk
import numpy as np
import torch
from tqdm.auto import tqdm


def configure_seed_logging(SEED=1873337):
    """
    Configure seed for reproducibility, configure logging messages
    Download nltk required libraries
    """
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.WARNING)


def save_pickle(save_to, save_what):
    """ Save data into pickle format"""
    with open(save_to, mode='wb') as f:
        pickle.dump(save_what, f)


def load_pickle(load_from):
    """ Load data from pre-saved pickle file"""
    with open(load_from, 'rb') as f:
        return pickle.load(f)


def ensure_dir(path):
    """
    Makes sure dir exists and if not it creates it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def compute_epoch_time(elapsed):
    """ Calculate epoch time in hh:mm:ss"""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_pretrained_embeddings(file_name, word2idx, embeddings_size, save_to=None):
    if os.path.exists(save_to):
        pretrained_embeddings = torch.from_numpy(np.load(save_to))

    else:
        fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}
        for line in tqdm(fin, desc=f'Reading data from {file_name}'):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(tokens[1:], dtype=np.float)

        pretrained_embeddings = torch.randn(len(word2idx), embeddings_size)
        initialised = 0
        for idx, word in enumerate(data):
            if word in word2idx:
                initialised += 1
                vector_ = torch.from_numpy(data[word])
                pretrained_embeddings[word2idx.get(word)] = vector_

        pretrained_embeddings[word2idx["<PAD>"]] = torch.zeros(embeddings_size)
        pretrained_embeddings[word2idx["<UNK>"]] = torch.zeros(embeddings_size)
        print(f'Loaded {initialised} vectors and instantiated random embeddings for {len(word2idx) - initialised}')

        np.save(save_to, pretrained_embeddings)  # save the file as "outfile_name.npy"
    return pretrained_embeddings


def parse_multilingual_embeddings(dir_path, word2idx, save_to=None, embeddings_size=300):
    """
    Takes dir_path (str) which points to the directory where all the pretrained
    multilingual embeddings files are located, reading one at a time creating a
    tensor of tensors with all the words in the vocab dict
    """
    if os.path.exists(save_to):
        pretrained_embeddings = torch.from_numpy(np.load(save_to))
    else:
        data = {}
        for path_ in glob.glob(dir_path):
            with open(path_, encoding='UTF-8', mode='r') as embeddings_file:
                for line in tqdm(embeddings_file, leave=False):
                    tokens = line.rstrip().split(' ')
                    data[tokens[0]] = np.array(tokens[1:], dtype=np.float)

        pretrained_embeddings = torch.randn(len(word2idx), embeddings_size)
        initialised = 0
        for idx, word in tqdm(enumerate(data), leave=False,
                              desc="Fetching pretrained embeddings as per our vocab"):
            if word in word2idx:
                initialised += 1
                vector_ = torch.from_numpy(data[word])
                pretrained_embeddings[word2idx.get(word)] = vector_

        pretrained_embeddings[word2idx["<PAD>"]] = torch.zeros(embeddings_size)
        pretrained_embeddings[word2idx["<UNK>"]] = torch.zeros(embeddings_size)
        pretrained_embeddings[word2idx["<SEP>"]] = torch.zeros(embeddings_size)
        print(f'Loaded {initialised} vectors and instantiated random embeddings for {len(word2idx) - initialised}')
        np.save(save_to, pretrained_embeddings)  # save the file as "outfile_name.npy"
    return pretrained_embeddings
