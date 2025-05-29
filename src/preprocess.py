# PREPARACIÓN DE LOS DATOS

import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path='data/reviews.csv'):
    df = pd.read_csv(path)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)


# CARGO LOS EMBEDDINGS PREENTRENADOS (Word2Vec)

from gensim.models import KeyedVectors

def load_embeddings(path='data/GoogleNews-vectors-negative300.bin'):
    return KeyedVectors.load_word2vec_format(path, binary=True)
