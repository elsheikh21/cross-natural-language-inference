import nltk
from transformers import AutoTokenizer, AutoModel
import gc


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
model = AutoModel.from_pretrained(model_name)
del model, tokenizer
gc.collect()
