import logging
import os

import torch
from data_loader import (NLPDatasetParser, K_NLPDatasetParser, BERTDatasetParser, read_train, read_dev, read_test)
from evaluater import compute_metrics
from models import BaselineModel, K_Model, HyperParameters, BERTModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from train import Trainer, WriterTensorboardX, BERT_Trainer
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
    is_bert = True

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "model", "word_stoi.pkl")
    configure_seed_logging()

    data = read_train()
    train_dataset = BERTDatasetParser(device_, data)
    train_dataset.encode_dataset()
    logging.info("Parsed and indexed training dataset")

    dev_data = read_dev()
    dev_dataset = BERTDatasetParser(device_, dev_data)
    dev_dataset.encode_dataset()
    logging.info("Parsed and indexed validation dataset")

    test_data = read_test()
    test_dataset = BERTDatasetParser(device_, test_data)
    test_dataset.encode_dataset()
    logging.info("Parsed and indexed testing dataset")

    # Set Hyperparameters
    pretrained_embeddings_ = None
    batch_size = 16

    name_ = 'mBERT'
    hp = HyperParameters(name_, None, train_dataset.label2idx, pretrained_embeddings_, batch_size)
    hp._print_info()

    # Prepare data loaders
    train_dataset_ = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                collate_fn=BERTDatasetParser.pad_batch,
                                shuffle=True)
    dev_dataset_ = DataLoader(dataset=dev_dataset,
                              batch_size=batch_size,
                              collate_fn=BERTDatasetParser.pad_batch)
    test_dataset_ = DataLoader(dataset=test_dataset,
                               batch_size=batch_size,
                               collate_fn=BERTDatasetParser.pad_batch)

    # Create and train model
    model = BERTModel(hp).to(device_)
    model.print_summary()

    log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
    writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)
    optimizer_ = AdamW(model.parameters(), lr=2e-5) if is_bert else Adam(model.parameters())
    epochs_ = 1

    trainer = BERT_Trainer(model=model, writer=writer_,
                           epochs=epochs_, _device=device_, verbose=True,
                           loss_function=CrossEntropyLoss(),
                           optimizer=optimizer_)

    # Either to train model from scratch or load a pretrained model
    model_save_to = os.path.join(os.getcwd(), 'model', f"{model.name}_model")
    if to_train:
        try:
            _ = trainer.train(train_dataset_, dev_dataset_, save_to=model_save_to)
        except (KeyboardInterrupt, RuntimeError):
            logging.warning("You halted the training process, saving the model and its weights...")
            model.save_(model_save_to)
    else:
        model.load_(f"{model_save_to}.pth")
        logging.info(f"Model loaded from {model_save_to}.pth")

    languages, predicted_labels = [], []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(test_dataset_, desc='Predicting on Testing dataset',
                           leave=False, total=len(test_dataset_)):
            languages.extend(sample["languages"])
            seq = sample["premises_hypotheses"].to(device_)
            mask = (seq != 0).to(device_, dtype=torch.uint8)
            labels = sample["outputs"].to(device_)
            token_types = sample["token_types"].to(device_)
            batch_predicted_labels = model.predict_sentence_(seq, mask, token_types)
            predicted_labels.extend(batch_predicted_labels)

    decoded_predicted_labels = [train_dataset.idx2label.get(tag_) for tag in predicted_labels for tag_ in tag]
    assert len(decoded_predicted_labels) == len(test_dataset.labels)
    compute_metrics(languages, decoded_predicted_labels, test_dataset.labels)
