# Scanner Training

## Models to Try

### LSTM
 [A Beginner’s Guide on Sentiment Analysis with RNN](https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e)

 ### RNN
 [RNNs For Keras Text Classification Tasks](https://coderzcolumn.com/tutorials/artificial-intelligence/keras-rnns-for-text-classification-tasks)

 ### CNN
 [CNNs With Keras Conv1D For Text Classification Tasks](https://coderzcolumn.com/tutorials/artificial-intelligence/keras-cnn-with-conv1d-for-text-classification)
 
 ### BERT
 [Classify text with BERT](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)

---
## Dataset Context
### Reddit dataset (train-balanced-sarcasm.csv)
Current code in the `Hartzler/*.ipynb` files removes the columns containing the **parent comment** text. You could instead combine the **comment** and **parent comment** columns into a single line of text and feed that through the model.
### Mustard++
Current code in the `Hartzler/*.ipynb` files removes the rows with dialouge that lead up to the final, labeled statement in the dialouge. You could instead build one long line of text using the previous dialouge combined with the final labeled statement so that its one big block of text for each sarastic/nonsarcastic label.
### iSarcasm (dataset_csv.csv)
Not much to do here. Pretty clean dataset
### NewsHeadlines (x1.json)
---

## Text Vectorization
- [Try Embeddings from Language Models (ELMo)](https://www.geeksforgeeks.org/overview-of-word-embedding-using-embeddings-from-language-models-elmo)

- [Test TextVectorization layer without stripping punctuation and/or lowercasing words](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization)

---

## Saving the Model

### Overview
https://keras.io/guides/serialization_and_saving/

### Saving the "best" version during the training
https://keras.io/api/callbacks/model_checkpoint/

## Keras API Docs
https://www.tensorflow.org/api_docs/python/tf/keras
