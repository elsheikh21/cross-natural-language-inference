import logging
import os

import torch
from data_loader import (NLPDatasetParser, K_NLPDatasetParser, read_train, read_dev, read_test)
from evaluater import compute_metrics
from models import BaselineModel, K_Model, HyperParameters
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from train import Trainer, WriterTensorboardX
from transformers import AdamW
from utils import configure_seed_logging


def print_summary(summary_data_, verbose=False):
    premises, dev_premises, test_premises, word2idx, label2idx = summary_data_
    if verbose:
        print("\n=============Data Summary======================",
              f"train_x length: {len(premises)} sentences",
              f"dev_x length: {len(dev_premises)} sentences",
              f"test_x length: {len(test_premises)} sentences",
              f"Vocab size: {len(word2idx)}",
              f"Labels vocab size: {len(label2idx)}",
              "===============================================\n", sep="\n")


if __name__ == "__main__":
    to_train = True
    k_format = True
    is_bert = True

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "model", "word_stoi.pkl")
    configure_seed_logging()

    data = read_train()
    _, premises, _, _ = data
    train_dataset = K_NLPDatasetParser(device_, data, is_bert) if k_format else NLPDatasetParser(device_, data, is_bert)
    word2idx, idx2word, label2idx, idx2label = train_dataset.create_vocabulary(save_path)
    train_dataset.encode_dataset(word2idx, label2idx)
    logging.info("Parsed and indexed training dataset")

    dev_data = read_dev()
    _, dev_premises, _, _ = dev_data
    dev_dataset = K_NLPDatasetParser(device_, dev_data, is_bert) if k_format else NLPDatasetParser(device_, dev_data, is_bert)
    dev_dataset.encode_dataset(word2idx, label2idx)
    logging.info("Parsed and indexed validation dataset")

    test_data = read_test()
    _, test_premises, _, _ = test_data
    test_dataset = K_NLPDatasetParser(device_, test_data, is_bert) if k_format else NLPDatasetParser(device_, test_data, is_bert)
    test_dataset.encode_dataset(word2idx, label2idx)
    logging.info("Parsed and indexed testing dataset")

    summary_data = premises, dev_premises, test_premises, word2idx, label2idx
    print_summary(summary_data)

    # Set Hyperparameters
    pretrained_embeddings_ = None
    batch_size = 16

    name_ = 'K_model'
    hp = HyperParameters(name_, word2idx, train_dataset.label2idx, pretrained_embeddings_, batch_size)
    # hp._print_info()

    # Prepare data loaders
    if k_format:
        train_dataset_ = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    collate_fn=K_NLPDatasetParser.pad_batch,
                                    shuffle=True)
        dev_dataset_ = DataLoader(dataset=dev_dataset,
                                  batch_size=batch_size,
                                  collate_fn=K_NLPDatasetParser.pad_batch)
        test_dataset_ = DataLoader(dataset=test_dataset,
                                   batch_size=batch_size,
                                   collate_fn=K_NLPDatasetParser.pad_batch)
    else:
        train_dataset_ = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    collate_fn=train_dataset.pad_batch,
                                    shuffle=True)
        dev_dataset_ = DataLoader(dataset=dev_dataset,
                                  batch_size=batch_size,
                                  collate_fn=dev_dataset.pad_batch)
        test_dataset_ = DataLoader(dataset=test_dataset,
                                   batch_size=batch_size,
                                   collate_fn=NLPDatasetParser.pad_batch)

        # Create and train model

    model = K_Model(hp).to(device_) if k_format else BaselineModel(hp).to(device_)
    # model.print_summary()

    log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
    writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)
    optimizer_ = AdamW(model.parameters(), lr=2e-5) if is_bert else Adam(model.parameters())
    epochs_ = 1

    trainer = Trainer(model=model, writer=writer_, verbose=True,
                      loss_function=CrossEntropyLoss(),
                      optimizer=optimizer_, epochs=epochs_,
                      _device=device_, is_k_format=k_format)

    # Either to train model from scratch or load a pretrained model
    save_to_ = os.path.join(os.getcwd(), 'model', f"{model.name}_model")
    if to_train:
        try:
            _ = trainer.train(train_dataset_, dev_dataset_, save_to=save_to_)
        except (KeyboardInterrupt, RuntimeError):
            logging.warning("You halted the training process, saving the model and its weights...")
            model.save_(save_to_)
    else:
        model.load_(f"{save_to_}.pth")
        logging.info(f"Model loaded from {save_to_}.pth")

    languages, predicted_labels = [], []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(test_dataset_, desc='Predicting on Testing dataset',
                           leave=False, total=len(test_dataset_)):
            languages.extend(sample["languages"])
            premises_seq = sample["premises"].to(device_)
            hypotheses_seq = sample["hypotheses"].to(device_)
            batch_predicted_labels = model.predict_sentence_(premises_seq, hypotheses_seq)
            predicted_labels.extend(batch_predicted_labels)

    decoded_predicted_labels = [train_dataset.idx2label.get(tag_) for tag in predicted_labels for tag_ in tag]
    # decoded_predicted_labels = K_NLPDatasetParser.decode_predictions(predicted_labels, label2idx)
    assert len(decoded_predicted_labels) == len(test_dataset.labels)
    compute_metrics(languages, decoded_predicted_labels, test_dataset.labels)
