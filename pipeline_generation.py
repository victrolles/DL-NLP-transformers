# Importation de la librairie transformers
from transformers import pipeline

# Définition de la phrase à compléter
sentense = "I can teach you how to"

# Création d'un objet nlp qui est un pipeline de génération de texte
generator = pipeline("text-generation")

# Utilisation de l'objet nlp pour générer du texte
result = generator(sentense, max_length=30, do_sample=True)

print("\nSentense: ", sentense)
print(result, "\n")
# Sentense:  I can teach you how to
# [{'generated_text': 'I can teach you how to move on; I can instruct you how to live your life.'}]