import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers

def create_and_compile_ga_model(max_tokens, num_classes, train_sentences):
    text_vectorizer = TextVectorization(
        max_tokens=max_tokens, output_sequence_length=55
    )
    text_vectorizer.adapt(train_sentences)

    rct_20k_text_vocab = text_vectorizer.get_vocabulary()

    token_embed = layers.Embedding(
        input_dim=len(rct_20k_text_vocab),
        output_dim=128,
        mask_zero=True,
        name="token_embedding",
    )

    inputs = layers.Input(shape=(1,), dtype=tf.string)
    text_vectors = text_vectorizer(inputs)
    token_embeddings = token_embed(text_vectors)

    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(
        token_embeddings
    )
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model

def create_and_compile_pretrained_embedding_model(tf_hub_embedding_layer, num_classes):
    inputs = layers.Input(shape=[], dtype=tf.string)
    pretrained_embedding = tf_hub_embedding_layer(inputs)
    x = layers.Dense(128, activation="relu")(pretrained_embedding)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    return model

def build_conv1D_char_embedding_model(char_vectorizer, char_embed, num_classes):
    inputs = layers.Input(shape=(1,), dtype="string")
    char_vectors = char_vectorizer(inputs)
    char_embeddings = char_embed(char_vectors)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(char_embeddings)
    x = layers.GlobalMaxPool1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model_3_conv1D_char_embedding")
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    return model

def build_token_char_hybrid_model(token_model, char_vectorizer, char_embed, num_classes):
    # Token model
    token_inputs = token_model.input
    token_output = token_model.output

    # Character model
    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
    char_vectors = char_vectorizer(char_inputs)
    char_embeddings = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(25))(char_embeddings)
    char_model = tf.keras.Model(inputs=char_inputs, outputs=char_bi_lstm)

    # Concatenate token and character embeddings
    token_char_concat = layers.Concatenate(name="token_char_hybrid")([token_output, char_model.output])

    # Additional layers 
    combined_dropout = layers.Dropout(0.5)(token_char_concat)
    combined_dense = layers.Dense(200, activation="relu")(combined_dropout)
    final_dropout = layers.Dropout(0.5)(combined_dense)
    output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)

    model = tf.keras.Model(inputs=[token_inputs, char_model.input], outputs=output_layer, name="model_4_token_and_char_embeddings")
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    return model
