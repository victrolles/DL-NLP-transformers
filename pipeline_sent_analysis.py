# Importation de la librairie transformers
from transformers import pipeline

sentence1 = "I love learning new things like NLP"
sentence2 = "I hate riding a bike when it's raining and windy"

# Load the pipeline
nlp = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Run the pipeline
result1, result2 = nlp(sentence1), nlp(sentence2)

print(sentence1, " : ", result1, "\n", sentence2, " : ", result2)
# I love learning new things like NLP  :  [{'label': 'POSITIVE', 'score': 0.9993341565132141}] 
# I hate riding a bike when it's raining and windy  :  [{'label': 'NEGATIVE', 'score': 0.9847230911254883}]