# Importation de la librairie transformers
from transformers import pipeline

sentence = "you are the best person"

# translate from English to French
nlp = pipeline(task="translation_en_to_fr", model="t5-base")

# Run the pipeline
result = nlp(sentence)
print(result)