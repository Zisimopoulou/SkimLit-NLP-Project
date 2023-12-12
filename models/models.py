import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers

def create_and_compile_ga_model(max_tokens, num_classes):
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

def create_and_compile_text_classification_model(tf_hub_embedding_layer, num_classes):
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
