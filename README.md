# Natural Language Inference (NLI)

- This project's scope is to build a ZeroShot Cross-lingual Natural Language Inference (XNLI). NLI is a pair sequences classification task, to classify the logical relationship between 2 sentences, whether it is entailment, contradiction, neutral.
- Data sets provided for this task where loaded using `Huggingface nlp` using both `MNLI` Multi-genre NLI English corpus for both training and validation data sets, and using cross-lingual NLI corpus `XNLI` made up of 15 languages (spanning over different language families).
    - dataset did not suffer any labels imbalance
    - data sets parser is found in `hw3/stud/data_loader` directory, different parsers were created for different models used
- Different models were used starting from BiLSTM model up to using Pretrained language models 
    1. Baseline model
    2. K_model
    3. mBERT
    4. XLM model based on either (XLM) or (XLM & TLM)
    5. XLM-RoBERTa
    - Models can be found in `hw3/stud/models/model_architectures.py` and hyperparameters script is in the same directory

---

To use this repo
- preferably create virtual env
```
conda create -n xnli_task python=3.7
conda activate xnli_task
```

- Install requirements

``` pip install -r requirements.txt ```

- To train the models

```
python3 hw3/stud/main_baseline.py  # for baseline or K model
python3 hw3/stud/main_bert.py  # for mBERT model
python3 hw3/stud/main_xlm.py  # for XLM model
python3 hw3/stud/main_xlm-r.py  # for XLM-R model 
```

- Check `XNLI_XLMR.ipynb` notebook for commented code, added to the scripts
- Check report for more theoritical insights `report.pdf`
