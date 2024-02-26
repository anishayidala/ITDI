import os 
import json
import pandas as pd
import re
from tqdm import tqdm
import numpy as np

print("Generating the dataset...")

# list of dialects in the training set
dialects = [d for d in os.listdir(".") if not os.path.isfile(d) and d != '.ipynb_checkpoints']

# define dictionaries for string->int and int->label conversion
fold_label = {
    'eml_texts' : 0,
    'nap_texts' : 1,
    'pms_texts' : 2,
    'fur_texts' : 3,
    'lld_texts' : 4,
    'lij_texts' : 5,
    'lmo_texts' : 6,
    'roa_tara_texts' : 7,
    'scn_texts' : 8,
    'vec_texts' : 9,
    'sc_texts' : 10,
    'it_texts' : 11
}
dial_label = {
    0 : 'EML',
    1 : 'NAP',
    2 : 'PMS',
    3 : 'FUR',
    4 : 'LLD',
    5 : 'LIJ',
    6 : 'LMO',
    7 : 'ROA_TARA',
    8 : 'SCN', 
    9 : 'VEC',
    10 : 'SC',
    11 : 'ITA'
}

# create training dataset
data = []

for d in tqdm(dialects):
    for name in os.listdir(d + "/AA/"):
        f = open(d + "/AA/" + name, "r")
        lines = f.readlines()
        for l in lines:
            jline = json.loads(l)
            if not jline['text']:
                continue
            data.append([int(jline['id']), jline['url'], jline['title'], jline['text'], fold_label[d]])


columns = {'id':int(), 'url':str, 'title':str, 'text':str, 'label':int()}
df = pd.DataFrame(data, columns = columns)

df = df.drop(columns=["id", "url", "title"])

# clean text
def clean(text):
    text = re.sub(r'==.*?==+', '', text)

    text = text.replace("\n", " ")

    text = text.replace('"', " ")

    regex = re.compile('&[^;]+;') 
    text = re.sub(regex, '', text)


    regex = re.compile('(graph.*/graph|\(.*\)|\[.*\]|parentid>.*/parentid>|BR[^>]+>|bR[^>]+>|Br[^>]+>|br[^>]+>|ns>.*/ns>|timestamp>.*/timestamp>|revision>.*/revision>|contributor>.*/contributor>|model>.*/model>|format>.*/format>|comment>.*/comment>)') 
    text = re.sub(regex, '', text)
    regex = re.compile('(parentid.*/parentid|ns.*/ns|timestamp.*/timestamp|revision.*/revision|contributor.*/contributor|model.*/model|format.*/format|comment.*/comment)') 
    text = re.sub(regex, '', text)

    text = text.replace("revision>", "")
    text = text.replace("br>", "")
    text = text.replace("Br>", "")
    text = text.replace("bR>", "")
    text = text.replace("BR>", "")
    text = text.replace("/br>", "")
    text = text.replace("/Br>", "")
    text = text.replace("/bR>", "")
    text = text.replace("/BR>", "")

    text = text.replace("&quot;","")

    text = text.replace("br clear=all>", "")

    if(len(text) < 50):
        text = np.nan

    return text

# print("Saving uncleaned dataset...")
# df.to_csv("uncleaned.csv", index=None)

print("Cleaning text...")

df['text'] = df['text'].apply(clean)

# drop rows with nan values
df.dropna(inplace=True)

# drop duplicate entries in the samples
df.drop_duplicates(subset ='text',keep = False, inplace = True) 

# create sentences
print("Splitting sentences...")

import spacy

nlp = spacy.load("it_core_news_sm", disable=['ner', 'lemmatizer', "textcat", "custom", "tagger"])

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, verbose=0)

df['text'] = df['text'].parallel_apply(nlp)

X = df["text"].to_numpy()           
y = df["label"].to_numpy()

print("Creating new data...")
X_train = []
y_train = []
for i, article in tqdm(enumerate(X), total=X.shape[0]):
  for sentence in article.sents:
    X_train.append(sentence)
    y_train.append(y[i])
X_train = np.array(X_train, dtype=object)
y_train = np.array(y_train, dtype=object)

print("Cleaning sentences...")
df = pd.DataFrame({'text': X_train, 'label': y_train}, index=None)

df["text"] = df['text'].apply(lambda x: ''.join(x.text))

# pms documents have a lot of these
df["text"] = df['text'].apply(lambda x: x.replace("http://www.sil.org/iso639-3/documentation.asp?id=", ""))
# other minor corrections
df['text'] = df['text'].apply(lambda x: x.replace("&lt;br clear=all&gt;", ""))
df['text'] = df['text'].apply(lambda x: x.replace("Evulusiù demogràfica.", ""))
df['text'] = df['text'].apply(lambda x: x.replace("&lt;br&gt;&lt;br&gt;", ""))
df['text'] = df['text'].apply(lambda x: x.replace("ł", "l"))

df['text'] = df['text'].apply(lambda x: x.replace("&lt;br clear=all&gt;", ""))
df['text'] = df['text'].apply(lambda x: x.replace("Evulusiù demogràfica.", ""))
df['text'] = df['text'].apply(lambda x: x.replace("&lt;br&gt;&lt;br&gt;", ""))
df['text'] = df['text'].apply(lambda x: x.replace("ł", "l"))
df['text'] = df['text'].apply(lambda x: x.replace("Ł", "l"))

df["text"] = df['text'].apply(lambda x: np.nan if len(x)<=20 else x)
df.dropna(inplace=True)

df.loc[df['label'] == 2, 'text'] = df.loc[df['label'] == 2, 'text'].apply(lambda x: np.nan if ("grup ëd popolassion." in x or "A confin-a con " in x or "a l’é na comun-a ëd" in x or "con na densità" in x or "A së stend" in x or "As dëstend për" in x or "a l'é na comun" in x or "La lenga" in x or "Në schema" in x or "Ël sìndich a l'é" in x or "a l'é un comun" in x) else x)
df.loc[df['label'] == 6, 'text'] = df.loc[df['label'] == 6, 'text'].apply(lambda x: np.nan if ("La Stazzion de" in x or "El cumün" in x or "a l'è una cità" in x or "El Passaport" in x or "la se tróa a 'na" in x or "a l'è 'na ferrovia" in x or "L'è taccada a stazione di" in x or "La a l'è 'na strada" in x or "L'andament del numer de abitant" in x or "L'andament del nömer dei abitàncc" in x or "l'è menziunaa la prima volta" in x or "l'è 'na stazion de la" in x or "L'andamènt del nömer dei abitàncc" in x or "La Stazion de" in x or "El Distret" in x or "El cümü" in x or "km²" in x or "Al gh’ha pressapoch abitant" in x or "l'è un cumün" in x or "El cumün de" in x or "El cunfìna coi cümü" in x or "l'è un cümü" in x or "l'è 'n cümü" in x or "e 'na densità de" in x) else x)
df.loc[df['label'] == 9, 'text'] = df.loc[df['label'] == 9, 'text'].apply(lambda x: np.nan if ("el xe on comun de" in x or "el xe un comun" in x or "gregorian" in x) else x)

df.dropna(inplace=True)
df.drop_duplicates(subset ='text', keep = False, inplace = True) 

df.to_csv("train.csv", index=None)

print("Dataset created.")