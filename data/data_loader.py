import tensorflow as tf
import pandas as pd
import os
import numpy as np
import random
import string
import tensorflow_hub as hub
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

def load_and_preprocess_pubmed_data(data_dir):    
    data_dir = os.path.join(data_dir, "pubmed-rct", "PubMed_20k_RCT_numbers_replaced_with_at_sign")

    train_samples = preprocess_text_with_line_numbers(os.path.join(data_dir, "train.txt"))
    val_samples = preprocess_text_with_line_numbers(os.path.join(data_dir, "dev.txt"))
    test_samples = preprocess_text_with_line_numbers(os.path.join(data_dir, "test.txt"))

    train_df = pd.DataFrame(train_samples)
    val_df = pd.DataFrame(val_samples)
    test_df = pd.DataFrame(test_samples)

    train_sentences = train_df["text"].tolist()
    val_sentences = val_df["text"].tolist()
    test_sentences = test_df["text"].tolist()

    return train_df, val_df, test_df, train_sentences, val_sentences, test_sentences

def encode_labels_one_hot(encoder, labels):
    one_hot_encoded_labels = encoder.transform(labels.reshape(-1, 1))
    return one_hot_encoded_labels

def get_lines(filename):
    with open(filename, "r") as f:
        return f.readlines()

def preprocess_text_with_line_numbers(filename):
  input_lines = get_lines(filename)  
  abstract_lines = ""  
  abstract_samples = [] 
  
  for line in input_lines:
    if line.startswith("###"):  
      abstract_id = line
      abstract_lines = ""  
    elif line.isspace():  
      abstract_line_split = abstract_lines.splitlines()  

      for abstract_line_number, abstract_line in enumerate(abstract_line_split):
        line_data = {}  
        target_text_split = abstract_line.split("\t")  
        line_data["target"] = target_text_split[0]  
        line_data["text"] = target_text_split[1].lower()  
        line_data["line_number"] = abstract_line_number  
        line_data["total_lines"] = len(abstract_line_split) - 1  
        abstract_samples.append(line_data)  
    
    else:  
      abstract_lines += line
  
  return abstract_samples


def perform_one_hot_encoding(one_hot_encoder, train_df, val_df, test_df, target_column="target"):
    train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
    val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
    test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))
    return train_labels_one_hot, val_labels_one_hot, test_labels_one_hot

def perform_label_encoding(label_encoder, train_df, val_df, test_df, target_column="target"):
    train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
    val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
    test_labels_encoded =label_encoder.transform(test_df["target"].to_numpy())
    return train_labels_encoded, val_labels_encoded, test_labels_encoded

def create_tf_datasets(sentences, labels_one_hot, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((sentences, labels_one_hot))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def split_chars(text):
  return " ".join(list(text))

def split_chars_in_sentences(sentences):
    return [split_chars(sentence) for sentence in sentences]

def create_char_vectorizer(train_sentences, output_percentile=95):
    char_lens = [len(sentence) for sentence in train_sentences]
    output_seq_char_len = int(np.percentile(char_lens, output_percentile))

    alphabet = string.ascii_lowercase + string.digits + string.punctuation
    NUM_CHAR_TOKENS = len(alphabet) + 2  # num characters in alphabet + space + OOV token

    # Create TextVectorization layer for character tokenization
    char_vectorizer = TextVectorization(
        max_tokens=NUM_CHAR_TOKENS,
        output_sequence_length=output_seq_char_len,
        standardize="lower_and_strip_punctuation",
        name="char_vectorizer"
    )
    char_vectorizer.adapt(train_sentences)
    return char_vectorizer


