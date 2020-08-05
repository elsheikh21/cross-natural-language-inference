import logging
import os

import nlp
import torch
from data_loader import (NLPDatasetParser, K_NLPDatasetParser, read_mnli, read_xnli)
from evaluater import compute_metrics
from models import BaselineModel, K_Model, BERT_Model, HyperParameters
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from train import Trainer, BERT_Trainer, WriterTensorboardX
from utils import configure_seed_logging


if __name__ == "__main__":
    # TODO: TEST DOCKER PIPELINE IMPLEMENTATION
    to_train = True
    k_format = True
    is_bert = True

    configure_seed_logging()
    data = read_mnli(nlp.load_dataset('multi_nli')['train'])
    languages, premises, hypotheses, labels = data

    dev_data = read_mnli(nlp.load_dataset('multi_nli')['validation_matched'])
    dev_lang, dev_premises, dev_hypotheses, dev_labels = dev_data

    test_data = read_xnli(nlp.load_dataset('xnli')['test'])
    test_lang, test_premises, test_hypotheses, test_labels = test_data

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "model", "word_stoi.pkl")

    train_dataset = K_NLPDatasetParser(device_, data, is_bert) if k_format else NLPDatasetParser(device_, data, is_bert)
    word2idx, idx2word, label2idx, idx2label = train_dataset.create_vocabulary(save_path)
    train_dataset.encode_dataset(word2idx, label2idx)
    logging.info("Parsed and indexed training dataset")

    dev_dataset = K_NLPDatasetParser(device_, dev_data, is_bert) if k_format else NLPDatasetParser(device_, dev_data,
                                                                                                   is_bert)
    dev_dataset.encode_dataset(word2idx, label2idx)
    logging.info("Parsed and indexed validation dataset")

    test_dataset = K_NLPDatasetParser(device_, test_data, is_bert) if k_format else NLPDatasetParser(device_, test_data,
                                                                                                     is_bert)
    test_dataset.encode_dataset(word2idx, label2idx)
    logging.info("Parsed and indexed testing dataset")

    print("\n===============================================")
    print(f"train_x length: {len(premises)} sentences")
    print(f"dev_x length: {len(dev_premises)} sentences")
    print(f"test_x length: {len(test_premises)} sentences")
    print(f"Vocab size: {len(word2idx)}")
    print(f"Labels vocab size: {len(label2idx)}")
    print("===============================================\n")

    pretrained_embeddings_ = None
    # Set Hyperparameters
    batch_size = 32

    name_ = 'mBERT'
    hp = HyperParameters(name_, word2idx, label2idx, pretrained_embeddings_, batch_size)
    hp._print_info()

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
    if not is_bert:
        model = K_Model(hp).to(device_) if k_format else BaselineModel(hp).to(device_)
    else:
        model = BERT_Model(hp).to(device_)
    model.print_summary()

    # BERT TRAINING CELL
    log_path = os.path.join(os.getcwd(), 'runs', hp.model_name)
    writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)

    # IMPORTANT https://github.com/huggingface/transformers/issues/1328#issuecomment-534956703
    if is_bert:
        trainer = BERT_Trainer(model=model, writer=writer_,
                               epochs=5, _device=device_, verbose=True,
                               loss_function=CrossEntropyLoss(ignore_index=0),
                               optimizer=Adam(model.parameters(), lr=5e-5))
    else:
        trainer = Trainer(model=model, writer=writer_, verbose=True,
                          loss_function=CrossEntropyLoss(ignore_index=0),
                          optimizer=Adam(model.parameters()), epochs=5,
                          _device=device_, is_k_format=k_format)

    save_to_ = os.path.join(os.getcwd(), 'model', f"{model.name}_model")
    # Either to train model from scratch or load a pretrained model
    if to_train:
        try:
            _ = trainer.train(train_dataset_, dev_dataset_, save_to=save_to_)
        except KeyboardInterrupt:
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
            if not k_format and not is_bert:
                premises_seq = sample["premises"].to(device_)
                hypotheses_seq = sample["hypotheses"].to(device_)
                batch_predicted_labels = model.predict_sentence_(premises_seq, hypotheses_seq)
            elif is_bert:
                seq = sample["premises_hypotheses"].to(device_)
                mask = (seq != 0).to(device_, dtype=torch.uint8)
                batch_predicted_labels = model.predict_sentence_(seq, mask)
            predicted_labels.extend(batch_predicted_labels)
    decoded_predicted_labels = [idx2label.get(label) for tag in predicted_labels for label in tag]
    compute_metrics(languages, decoded_predicted_labels, test_labels)