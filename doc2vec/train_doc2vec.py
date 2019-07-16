import os
import glob

from bs4 import BeautifulSoup
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import MeCab
from tqdm import tqdm


def tokenize(text: str) -> list:
    tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    return tagger.parse(text).strip()


ROOT = 'extracted'
OUT_TEXT = 'sentences.txt'

text_dirs = glob.glob(os.path.join(ROOT, '*'))[:2]
text_paths = []
for text_dir in text_dirs:
    text_paths += glob.glob(os.path.join(text_dir, '*'))

documents = []
for t in tqdm(text_paths):
    with open(t, 'r', encoding='utf-8') as f:
        texts = f.read()

    soup = BeautifulSoup(texts)
    for doc in soup.find_all('doc'):
        title = doc.get('title')
        text = doc.get_text()
        words = tokenize(text)
        documents.append(TaggedDocument(words, [title]))


settings = {
    'dbow300d': {'vector_size': 300,
                 'epochs': 20,
                 'window': 15,
                 'min_count': 5,
                 'dm': 0,  # PV-DBOW
                 'dbow_words': 1,
                 'workers': 8},
    'dmpv300d': {'vector_size': 300,
                 'epochs': 20,
                 'window': 10,
                 'min_count': 2,
                 'alpha': 0.05,
                 'dm': 1,  # PV-DM
                 'sample': 0,
                 'workers': 8}
}

for setting_name, setting in settings.items():
    model = Doc2Vec(documents=documents, **setting)
    model.save(f'jawiki.doc2vec.{setting_name}.model')
    model.save_word2vec_format(f'jawiki.doc2vec.{setting_name}.model.bin',
                               doctag_vec=True,
                               prefix='ent_',
                               binary=True)