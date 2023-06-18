#Impootation de la librairie transformers
from transformers import pipeline

# Création d'un objet nlp qui est un pipeline de masquage
unmasker = pipeline('fill-mask', model='bert-base-uncased')

# Utilisation de l'objet nlp pour faire un masquage
result = unmasker("Hello I'm a [MASK] model.")

print(result)
sequences = [item['sequence'] for item in result]

# Print the sequences
for sequence in sequences:
    print(sequence)

# hello i'm a fashion model.
# hello i'm a role model.
# hello i'm a new model.
# hello i'm a super model.
# hello i'm a fine model.