from transformers import AutoTokenizer, AutoModelForMaskedLM

model_str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

tokenizer = AutoTokenizer.from_pretrained(model_str)

model = AutoModelForMaskedLM.from_pretrained(model_str)