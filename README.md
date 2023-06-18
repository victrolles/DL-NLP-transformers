# Rapport TX52
## NLP et développement de benchmarks
![image](https://github.com/victrolles/DL-NLP-transformers/assets/95492416/7989b389-7a18-4201-b0ec-35538d68833a)

**Ce github contient tous les codes python et notebook utilisées durant ce projet**
### Pipeline
Vous retrouverez tous les pipelines expérimentés :
 - classification : [pipeline_classification.py](https://github.com/victrolles/DL-NLP-transformers/blob/main/pipeline_classification.py)
 - Générateur de conversation : [pipeline_consersational.py](https://github.com/victrolles/DL-NLP-transformers/blob/main/pipeline_consersational.py)
 - Génération de texte : [pipeline_generation.py](https://github.com/victrolles/DL-NLP-transformers/blob/main/pipeline_generation.py)
 - Trouver le mot caché : [pipeline_mask.py](https://github.com/victrolles/DL-NLP-transformers/blob/main/pipeline_mask.py)
 - Analyse de sentiment : [pipeline_sent_analysis.py](https://github.com/victrolles/DL-NLP-transformers/blob/main/pipeline_sent_analysis.py)
 - Traducteur Anglais Français : [pipeline_translation.py](https://github.com/victrolles/DL-NLP-transformers/blob/main/pipeline_translation.py)

### Autres fonctionnalités de transformers
Code python pour découvrir la manipulation de modèle et tokenizer et des fonctions existantes :
 - [model_tokenizer.py](https://github.com/victrolles/DL-NLP-transformers/blob/main/model_tokenizer.py)

### Cours Nvidia
Notebook pour découvrir l'entrainement et le déployement de modèle sur Riva :
 - Entrainement d'un modèle : [text-classification-training.ipynb](https://github.com/victrolles/DL-NLP-transformers/blob/main/text-classification-training.ipynb)
 - Déployement d'un modèle : [text-classification-deployment.ipynb](https://github.com/victrolles/DL-NLP-transformers/blob/main/text-classification-deployment.ipynb)

### Affinage d'un modèle BERT
Ce notebook contient le code pour affiner un modèle et découvrir la librairie sklearn utilisé pour mesure les performances de modèle :
 - [FineTune_BERT_Model.ipynb](https://github.com/victrolles/DL-NLP-transformers/blob/main/FineTune_BERT_Model.ipynb)

### Benchmark de modèle HuggingFace
Ce notebook contient le code pour directement obtenir les résultats d'analyse (accuracy, precision, recall, f1) :
 - [BenchmarkNlpModels.ipynb](https://github.com/victrolles/DL-NLP-transformers/blob/main/BenchmarkNlpModels.ipynb)
