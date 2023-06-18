# Importation de la librairie transformers
from transformers import pipeline

# Création d'un objet nlp qui est un pipeline de classification
nlp = pipeline(task="zero-shot-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Utilisation de l'objet nlp pour faire une classification
result = nlp(
    "We need to change the way of working in the company to make more money",
    candidate_labels=["education", "politics", "business"],
)

print(result)