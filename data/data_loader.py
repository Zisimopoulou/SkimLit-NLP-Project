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
    """
    Clone the PubMed RCT GitHub repository, load and preprocess the data,
    and return train, validation, and test datasets.

    Args:
    - data_dir (str): Directory path for the PubMed RCT data.

    Returns:
    - train_sentences (list): List of sentences in the training set.
    - val_sentences (list): List of sentences in the validation set.
    - test_sentences (list): List of sentences in the test set.
    """
    data_dir = os.path.join(data_dir, "pubmed-rct", "PubMed_20k_RCT_numbers_replaced_with_at_sign")

    # Preprocess and load data
    train_samples = preprocess_text_with_line_numbers(data_dir, "train.txt")
    val_samples = preprocess_text_with_line_numbers(data_dir, "dev.txt")
    test_samples = preprocess_text_with_line_numbers(data_dir, "test.txt")
    
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
  input_lines = get_lines(filename) # get all lines from filename
  abstract_lines = "" # create an empty abstract
  abstract_samples = [] # create an empty list of abstracts
  
  # Loop through each line in target file
  for line in input_lines:
    if line.startswith("###"): # check to see if line is an ID line
      abstract_id = line
      abstract_lines = "" # reset abstract string
    elif line.isspace(): # check to see if line is a new line
      abstract_line_split = abstract_lines.splitlines() # split abstract into separate lines

      # Iterate through each line in abstract and count them at the same time
      for abstract_line_number, abstract_line in enumerate(abstract_line_split):
        line_data = {} # create empty dict to store data from line
        target_text_split = abstract_line.split("\t") # split target label from text
        line_data["target"] = target_text_split[0] # get target label
        line_data["text"] = target_text_split[1].lower() # get target text and lower it
        line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract?
        line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are in the abstract? (start from 0)
        abstract_samples.append(line_data) # add line data to abstract samples list
    
    else: # if the above conditions aren't fulfilled, the line contains a labelled sentence
      abstract_lines += line
  
    return abstract_samples

def perform_one_hot_encoding(train_df, val_df, test_df, target_column="target"):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
    val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
    test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))
    return train_labels_one_hot, val_labels_one_hot, test_labels_one_hot

def perform_label_encoding(train_df, val_df, test_df, target_column="target"):
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
    val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
    test_labels_encoded =label_encoder.transform(test_df["target"].to_numpy())
    return train_labels_encoded, val_labels_encoded, test_labels_encoded

def create_tf_datasets(sentences, labels_one_hot, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((sentences, labels_one_hot))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


