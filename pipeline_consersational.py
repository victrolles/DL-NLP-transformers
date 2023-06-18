# Importation de la librairie transformers
from transformers import pipeline, Conversation

# Création d'un objet nlp qui est un pipeline de conversation
nlp = pipeline(task="conversational", model="facebook/blenderbot-400M-distill")

# Initialisation de la conversation
conversation = Conversation(None)

# Boucle de conversation
while True:
    user_input = input("user >> ")
    if user_input == "quit":
        break
    conversation.add_user_input(user_input)
    bot_input = nlp(conversation)
    print("bot >>", str(bot_input))
print(conversation)


## Output:
## Conversation 1
## user >> Going to the movies tonight - any suggestions? 
## bot >> The Big Lebowski ,
## Conversation 2
## user >> What's the last book you have read? 
## bot >> The Last Question
