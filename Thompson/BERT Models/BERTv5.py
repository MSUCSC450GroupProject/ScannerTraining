# BERT model

import math
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

print(tf.__version__)

#tf.config.run_functions_eagerly(True)
tf.config.list_physical_devices('GPU')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


# Adjust sitcom data
data_sitcoms = pd.read_csv("../../shared_data/mustard++_text.csv")
data_sitcoms = data_sitcoms.drop(columns=['SCENE','KEY','END_TIME','SPEAKER','SHOW','Sarcasm_Type','Implicit_Emotion','Explicit_Emotion','Valence','Arousal'], axis=1)
data_sitcoms = data_sitcoms.rename(columns={'SENTENCE':'text','Sarcasm':'label'})
#print("data_sitcoms")
#print(data_sitcoms.info())

# Adjust tweets data
data_tweets = pd.read_csv("../../shared_data/dataset_csv.csv")
data_tweets = data_tweets.rename(columns={'tweets':'text'})
#print("data_tweets")
#print(data_tweets.info())

# Combine 2 datasets
data = pd.concat([data_tweets,data_sitcoms], ignore_index=True)

# remove non string and nan rows
for index, row in data.iterrows():
    if not type(row['text']) == str:
        data = data.drop(index, axis='index')
    if math.isnan(row['label']):
        data = data.drop(index, axis='index')

# Shuffle the rows
data = data.sample(frac=1).reset_index(drop=True)
print("combined data")
print(data.info())

subset_size = len(data['text'])
testing_size = int(subset_size * 0.4)
validation_size = int(subset_size * 0.2)
shuffle_size = subset_size - validation_size

data_batch_size = 32

data = data.sample(frac=1).reset_index(drop=True) ##was just data
train_data = data.head(subset_size) ##was just data
test_data = data.tail(testing_size) ##was just data

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

epochs = 150
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = 3e-5

#define the parameters for tokenizing and padding
vocab_size = 10000
embedding_dim = 32
max_length = 500

preprocessing_layer = hub.KerasLayer(
    'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3', 
    name='preprocessing'
)

bert_encoder = hub.KerasLayer(
    'https://tfhub.dev/tensorflow/bert_en_wwm_cased_L-24_H-1024_A-16/4', 
    trainable=True, 
    name='BERT_encoder'
)

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = preprocessing_layer(text_input)
    outputs = bert_encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

optimizer = optimization.create_optimizer(
    init_lr=init_lr,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    optimizer_type='adamw'
)

classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

saved_model_path = '../model_saves/bert_v5/'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=saved_model_path,
    monitor='val_binary_accuracy',
    mode='max',
    save_best_only=True
)

history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs,
                               callbacks=[checkpoint])

history_dict = history.history
print(history_dict.keys())

### Test the model
loss, accuracy = classifier_model.evaluate(test_ds.batch(data_batch_size)) ## change batch from 32 to 2

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
