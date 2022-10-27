import math
import spacy
import classy_classification
import pandas as pd

## NEED TO RUN
## python -m spacy download en_core_web_md

### Import the dataset

data_news_headlines = pd.read_json("../shared_data/x1.json")

# Adjust news headline data
data_news_headlines = data_news_headlines.drop(columns='article_link', axis=1)
data_news_headlines = data_news_headlines.rename(columns ={'headline':'text', 'is_sarcastic':'label'})
data_news_headlines = data_news_headlines.reindex(columns=['text','label'])
data_news_headlines.info()

### Set the dataset variables

subset_size = 2000
testing_size = int(subset_size * 0.2)

### Shuffle the data and set the train and test splits
data = data_news_headlines.sample(frac=1).reset_index(drop=True)
train_data = data.head(subset_size - testing_size)
test_data = data.tail(testing_size)

data ={"non-sarcastic":[],"sarcastic":[]}

for index, row in train_data.iterrows():
    if not math.isnan(row['label']):
        if bool(row['label']):
            data["sarcastic"].append(row['text'])
        else:
            data["non-sarcastic"].append(row['text'])

print("non-sarcastic", data["non-sarcastic"][:3])
print("sarcastic", data["sarcastic"][:3])

print("\nloading model")
sarc_detector = spacy.load("en_core_web_md")
print("\nadding text data")
sarc_detector.add_pipe(
    "text_categorizer",
    config={
        "data": data,
        "model": "spacy"
    }
)

print("\ntesting",sarc_detector("Please, keep talking. I always yawn when I am interested.")._.cats)