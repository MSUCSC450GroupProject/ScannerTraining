import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback

print(tf.__version__)

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

print(tf.config.list_physical_devices('GPU'))

### Import the dataset

data_news_headlines = pd.read_json("../shared_data/x1.json")

# Adjust news headline data
data_news_headlines = data_news_headlines.drop(columns='article_link', axis=1)
data_news_headlines = data_news_headlines.rename(columns ={'headline':'text', 'is_sarcastic':'label'})
data_news_headlines = data_news_headlines.reindex(columns=['text','label'])
data_news_headlines.info()

### Set the dataset variables

subset_size = len(data_news_headlines.index)
testing_size = int(subset_size * 0.2)
validation_size = testing_size
shuffle_size = subset_size - validation_size

data_batch_size = 32

### Shuffle the data and set the train and test splits

data = data_news_headlines.sample(frac=1).reset_index(drop=True)
train_data = data.head(subset_size - testing_size)
test_data = data.tail(testing_size)
train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_data['text'][validation_size:], 
        train_data['label'][validation_size:]
    )
).shuffle(shuffle_size).batch(data_batch_size)

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_data['text'][:validation_size],
        train_data['label'][:validation_size]
    )
).batch(data_batch_size)

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_data['text'],
        test_data['label']
    )
)

text_vocab_ds = tf.data.Dataset.from_tensor_slices(train_data['text'])

### Set training variables

epochs = 400

#define the parameters for tokenizing and padding
vocab_size = 10000
embedding_dim = 32
max_length = 120

### Initialize and config the Weights and Biases graphing library

wandb.init(project="sarcasmscanner", entity="awesomepossum")

wandb.config = {
    "epochs": epochs,
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "max_sentence_word_length": max_length,
    "batch_size": data_batch_size,
    "subset_size": subset_size,
    "training_size": subset_size - testing_size - validation_size,
    "testing_size": testing_size,
    "validation_size": validation_size,
    "dataset": "news_headlines",
    "architecture": "LSTM"
}

### Create the text vectorization layer and create the vocab

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, standardize='lower_and_strip_punctuation', split='whitespace', output_mode='int', output_sequence_length=max_length)

vectorize_layer.adapt(text_vocab_ds.batch(data_batch_size))


### Create the model
model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

### Train the model
model.fit(x=train_ds, validation_data=val_ds, epochs=epochs,callbacks=[WandbCallback()])

### Test the model
loss, accuracy = model.evaluate(test_ds.batch(32))

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

### Export for inference
saved_model_path = './model_saves/lstm_v3/'
model.save(saved_model_path, include_optimizer=False)
