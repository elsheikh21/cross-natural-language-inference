import logging
import os

import nlp
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_loader import NLPDatasetParser, read_mnli, read_xnli
from models import BaselineModel, HyperParameters
from train import Trainer, WriterTensorboardX
from utils import configure_seed_logging
from evaluater import compute_metrics


if __name__ == "__main__":
    configure_seed_logging()
    data = read_mnli(nlp.load_dataset('multi_nli')['train'])
    languages, premises, hypotheses, labels = data

    dev_data = read_mnli(nlp.load_dataset('multi_nli')['validation_matched'])
    dev_lang, dev_premises, dev_hypotheses, dev_labels = dev_data

    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), "model", "word_stoi.pkl")
    train_dataset = NLPDatasetParser(device_, data)
    word2idx, idx2word, label2idx, idx2label = train_dataset.create_vocabulary(save_path)
    train_dataset.encode_dataset(word2idx, label2idx)

    dev_dataset = NLPDatasetParser(device_, dev_data)
    dev_dataset.encode_dataset(word2idx, label2idx)

    print("\n===============================================")
    print(f"Train data_x length: {len(premises)} sentences")
    print(f"Dev data_x length: {len(dev_premises)} sentences")
    print(f"Vocab size: {len(word2idx)}")
    print(f"Labels vocab size: {len(label2idx)}")
    print("===============================================")

    # Set Hyper-parameters
    pretrained_embeddings_ = None
    batch_size = 32

    name_ = 'BiLSTM_Model'
    hp = HyperParameters(name_, word2idx, label2idx, pretrained_embeddings_, batch_size)
    hp.print_info()

    # Prepare data loaders
    train_dataset_ = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                collate_fn=NLPDatasetParser.pad_batch,
                                shuffle=True)
    dev_dataset_ = DataLoader(dataset=dev_dataset, batch_size=batch_size,
                              collate_fn=NLPDatasetParser.pad_batch)

    # Create and train model
    model = BaselineModel(hp).to(device_)
    model.print_summary()

    log_path = os.path.join(os.getcwd(), "runs", hp.model_name)
    writer_ = WriterTensorboardX(log_path, logger=logging, enable=True)

    trainer = Trainer(model=model, writer=writer_, verbose=True,
                      loss_function=CrossEntropyLoss(ignore_index=0),
                      optimizer=Adam(model.parameters()), epochs=1,
                      _device=device_)

    save_to_ = os.path.join(os.getcwd(), "model", f"{model.name}_model")
    try:
        _ = trainer.train(train_dataset_, dev_dataset_, save_to=save_to_)
    except KeyboardInterrupt:
        model.save_(save_to_)

    test_data = read_xnli(nlp.load_dataset('xnli')['test'])
    test_lang, test_premises, test_hypotheses, test_labels = test_data

    test_dataset = NLPDatasetParser(device_, test_data)
    test_dataset.encode_dataset(word2idx, label2idx)
    test_dataset_ = DataLoader(dataset=test_dataset, batch_size=batch_size,
                               collate_fn=NLPDatasetParser.pad_batch)

    languages, predicted_labels = [], []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(test_dataset_, desc='Predicting on Testing dataset',
                           leave=False, total=len(test_dataset_)):
            premises_seq, hypotheses_seq = sample["premises"].to(device_), sample["hypotheses"].to(device_)
            batch_predicted_labels = model.predict_sentence_(premises_seq, hypotheses_seq)
            languages.extend(sample["languages"])
            predicted_labels.extend(batch_predicted_labels)

    predicted_labels_ = [_e for e in predicted_labels for _e in e]
    decoded_predicted_labels = NLPDatasetParser.decode_predictions(predicted_labels, idx2label)
    compute_metrics(languages, decoded_predicted_labels, test_labels)
