import datetime
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

#tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

print(tf.config.list_physical_devices('GPU'))
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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

data_batch_size = 64

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

epochs = 500

#define the parameters for tokenizing and padding
vocab_size = 5000
embedding_dim = 32
max_length = 500


### Create the text vectorization layer and create the vocab

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, standardize='lower_and_strip_punctuation', split='whitespace', output_mode='int', output_sequence_length=max_length)

vectorize_layer.adapt(text_vocab_ds.batch(data_batch_size))


### Create the model
model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

### Model Saving, Logging, and Stopping
saved_model_path = './model_saves/lstm_v7/'
log_path = './logs/lstm_v7.csv'
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=saved_model_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)
logger = tf.keras.callbacks.CSVLogger(log_path)
class StopAtFivePM(tf.keras.callbacks.Callback):
    def on_epoch_begin(self):
        current_hour = datetime.datetime.now().hour
        if current_hour > 17:
            self.model.stop_training = True
early_stopper = StopAtFivePM()

### Train the model
model.fit(x=train_ds, validation_data=val_ds, epochs=epochs,callbacks=[save_checkpoint, logger, early_stopper])

### Test the model
loss, accuracy = model.evaluate(test_ds.batch(32))

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

### Export for inference
saved_model_path = './model_saves/lstm_v3/'
model.save(saved_model_path, include_optimizer=False)
