# BERT model

import math
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization
import wandb
from wandb.keras import WandbCallback

print(tf.__version__)

tf.config.run_functions_eagerly(True)
tf.config.list_physical_devices('GPU')

data_sitcoms = pd.read_csv("../shared_data/mustard++_text.csv")

# Adjust sitcom data
data_sitcoms = data_sitcoms.drop(columns=['SCENE','KEY','END_TIME','SPEAKER','SHOW','Sarcasm_Type','Implicit_Emotion','Explicit_Emotion','Valence','Arousal'], axis=1)
data_sitcoms = data_sitcoms.rename(columns={'SENTENCE':'text','Sarcasm':'label'})

# remove empty label rows
for index, row in data_sitcoms.iterrows():
    if math.isnan(row['label']):
        data_sitcoms = data_sitcoms.drop(index, axis='index')

data_tweets = pd.read_csv("../shared_data/dataset_csv.csv")

# Adjust tweets data
data_tweets = data_tweets.rename(columns={'tweets':'text'})
data_tweets.head()

# Combine all 4 datasets
#data = pd.concat([data_news_headlines,data_tweets,data_sitcoms,data_reddit], ignore_index=True)
# Combine 3 datasets
data = pd.concat([data_tweets,data_sitcoms], ignore_index=True)

# remove non string (nan) rows
for index, row in data.iterrows():
    if not type(row['text']) == str:
        data = data.drop(index, axis='index')

# Shuffle the rows
data = data.sample(frac=1).reset_index(drop=True)


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

### Initialize and config the Weights and Biases graphing library
w_config = {
    "epochs": epochs,
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "max_sentence_word_length": max_length,
    "batch_size": data_batch_size,
    "subset_size": subset_size,
    "training_size": subset_size - testing_size - validation_size,
    "testing_size": testing_size,
    "validation_size": validation_size,
    "dataset": "sitcoms",
    "architecture": "BERT"
}

wandb.init(project="sarcasmscanner", entity="awesomepossum", config=w_config)

preprocessing_layer = hub.KerasLayer(
    'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3', 
    name='preprocessing'
)

bert_encoder = hub.KerasLayer(
    'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/4', 
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

text_test = ["Please, keep talking. I always yawn when I am interested."]
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

optimizer = optimization.create_optimizer(
    init_lr=init_lr,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    optimizer_type='adamw'
)

classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs,
                               callbacks=[WandbCallback()])

history_dict = history.history
print(history_dict.keys())

### Test the model
loss, accuracy = classifier_model.evaluate(test_ds.batch(32)) ## change batch from 32 to 2

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']


saved_model_path = './model_saves/bert_v2/'
classifier_model.save(saved_model_path, include_optimizer=False)