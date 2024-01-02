# SkimLit-NLP-Project
In this project, we are implementing a Natural Language Processing (NLP) model based on the 2017 paper PubMed 200k RCT, aiming to classify sentences within medical abstracts into specific roles (e.g., objective, methods, results). The ultimate goal is to facilitate efficient literature review for researchers by allowing them to skim through abstracts and delve deeper when necessary, addressing the challenge posed by the increasing number of Randomized Controlled Trial (RCT) papers with unstructured abstracts. 

The project involves downloading the PubMed RCT200k dataset, preprocessing the data, conducting various modeling experiments, and building a multimodal model to replicate the architecture proposed in the referenced paper. Finally we choose the best-performing model model for our test data.

Kaggle was employed to utilize GPU capabilities for enhanced computational power.

The models:
    Global Average Model (Model 1):
        Custom token embeddings and Conv1D layers for text classification.
        Used custom functions for model creation, compilation, training, and evaluation.
        Employed a global average pooling strategy.

    Pre-trained Embedding Model (Model 2):
        Utilized a pre-trained embedding layer (tf_hub_embedding_layer) for text classification.
        Applied custom functions for model creation, compilation, training, and evaluation.

    Conv1D Character Embedding Model (Model 3):
        Implemented a Conv1D character embedding model for text classification.
        Used custom functions for model creation, compilation, training, and evaluation.

    Token and Character Hybrid Model (Model 4):
        Combined token and character embeddings with additional layers for classification.
        Used a hybrid model architecture with both token and character embeddings.

    Positional Token Character Embedding Model (Model 5):
        Incorporated positional information (line numbers and total lines) along with token and character embeddings.
        Created datasets combining one-hot encoded line numbers, total lines, sentences, and characters.

    Modified Trihybrid Model with Callbacks (Model 6):
        Enhanced the positional token character embedding model with callbacks for model checkpointing, early stopping, and learning rate reduction.
        Trained with specified callbacks for a dynamic training process.
        Employed a trihybrid model architecture with token, character, and positional embeddings.
