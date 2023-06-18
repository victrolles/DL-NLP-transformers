from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = tokenizer.model_max_length

inputs = ["I love learning about transformers and deep learning with the Hugging Face library.",
          "I don't like rugby because it's too violent for me."]

batch = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

print("\nsentences : ", inputs[0], "\n", inputs[1], "\n")
print("max_length : ", max_length, "\n")
print("batch : ", batch, "\n")

# batch :  {'input_ids': tensor([[  101,  1045,  2293,  4083,  2055, 19081,  1998,  2784,  4083,  2007,
#           1996, 17662,  2227,  3075,  1012,   102,     0],
#         [  101,  1045,  2123,  1005,  1056,  2066,  4043,  2138,  2009,  1005,
#           1055,  2205,  6355,  2005,  2033,  1012,   102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

with torch.no_grad():
    outputs = model(**batch)
    predictions = F.softmax(outputs.logits, dim=1)    
    labels = torch.argmax(predictions, dim=1)   
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]

print("outputs : ", outputs, "\n")
print("predictions : ", predictions, "\n")
print("labels : ", labels, "\n")
print("labels : ", labels, "\n")

# outputs :  SequenceClassifierOutput(loss=None, logits=tensor([[-3.5094,  3.7012],
#         [ 3.4061, -2.8610]]), hidden_states=None, attentions=None)

# predictions :  tensor([[7.3817e-04, 9.9926e-01],
#         [9.9811e-01, 1.8941e-03]])

# labels :  tensor([1, 0])

# labels :  ['POSITIVE', 'NEGATIVE']