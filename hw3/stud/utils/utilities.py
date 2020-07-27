import io
import logging
import os
import pickle
import random

import nltk
import numpy as np
import torch
from tqdm.auto import tqdm


def configure_seed_logging():
    """
    Configure seed for reproducibility, configure logging messages
    Download nltk required libraries
    """
    seed = 1873337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    nltk.download('punkt', quiet=True)
    torch.backends.cudnn.deterministic = True
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)


def save_pickle(save_to, save_what):
    """ Save data into pickle format"""
    with open(save_to, mode='wb') as f:
        pickle.dump(save_what, f)


def load_pickle(load_from):
    """ Load data from presaved pickle file"""
    with open(load_from, 'rb') as f:
        return pickle.load(f)


def ensure_dir(path):
    """
    Makes sure dir exists and if not it creates it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def compute_epoch_time(start_time, end_time):
    """ Calculate epoch time in minutes and seconds"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
