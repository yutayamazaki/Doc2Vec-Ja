# Doc2Vec-Ja
Doc2Vec: https://cs.stanford.edu/~quocle/paragraph_vector.pdf  
Details are available [here(Japanese)](https://scrapbox.io/whey-memo/%E6%97%A5%E6%9C%AC%E8%AA%9EDoc2Vec%E3%82%92Wiki%E3%82%B3%E3%83%BC%E3%83%91%E3%82%B9%E3%81%8B%E3%82%89%E4%BD%9C%E3%82%8B). 

## Usage

```shell
git clone https://github.com/yutayamazaki/Doc2Vec-Ja.git
cd Doc2Vec-Ja
# Activate Pipenv
pipenv shell
```

- Download latest wikipedia corpus jawiki-latest-pages-articles.xml.bz from [here](https://dumps.wikimedia.org/jawiki/) to Doc2Vec/doc2vec directory.
- Doanload WikiExtractor.py from https://github.com/attardi/wikiextractor to Doc2Vec/doc2vec directory.
- Then, Execute WikiExtractor.py to extract text from jawiki-latest-pages-articles.xml.bz.

```shell
cd doc2vec
python WikiExtractor.py
```

- Finally, train Doc2Vec model.

```shell
python train_doc2vec.py
```
