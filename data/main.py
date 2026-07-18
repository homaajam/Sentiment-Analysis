import pandas as pd
from source.preprocess import clean_text, tokenize


df= pd.read_csv("data/imdb.csv")
texts= [tokenize(clean_text(t)) for t in df["review"]]
labels= df["sentiment"].map({"positive":1, "negative":0}).tolist()

print(texts[0])
print(labels[0])