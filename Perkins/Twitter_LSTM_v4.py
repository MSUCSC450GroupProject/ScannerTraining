import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
tf.config.list_physical_devices('GPU')

data_tweets = pd.read_csv("../shared_data/dataset_csv.csv")

# Adjust tweets data
data_tweets = data_tweets.rename(columns={'tweets':'text'})
data_tweets.head()

subset_size = len(data_tweets.index)
testing_size = int(subset_size * 0.2)
validation_size = testing_size
shuffle_size = subset_size - validation_size

data_batch_size = 32

data = data_tweets.sample(frac=1).reset_index(drop=True)
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

epochs = 1000

##define the parameters for tokenizing and padding
vocab_size = 10000
embedding_dim = 32
max_length = 120

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, standardize='lower_and_strip_punctuation', split='whitespace', output_mode='int', output_sequence_length=max_length)

vectorize_layer.adapt(text_vocab_ds.batch(data_batch_size))

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


history = model.fit(x=train_ds,
                        validation_data=val_ds,
                        epochs=epochs)

loss, accuracy = model.evaluate(test_ds.batch(32))
print('\nTesting')
print(f'Testing Loss: {loss}')
print(f'Testing Accuracy: {accuracy}')

saved_model_path = './model_saves/twitter/'
model.save(saved_model_path, include_optimizer=False)