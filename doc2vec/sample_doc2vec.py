from gensim.models.doc2vec import Doc2Vec
import MeCab
from scipy.spatial.distance import cosine


def cosine_similarity(w, u):
    return 1 - cosine(w, u)


model = Doc2Vec.load('jawiki.doc2vec.dbow300d.model')
tagger = MeCab.Tagger('-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

texts = [
    '世界一のビッグクラブ、レアル・マドリードの一員になった日本代表MF久保建英(18)に対する評価が、ピッチの内外で急上昇している新シーズンはBチームのレアル・マドリード・カスティージャに登録される予定の久保は現在、カナダ・モントリオールで行われているトップチームのサマーキャンプに参加している。トップチームを率いるジネディーヌ・ジダン監督(47)の意向と言われるなかで、豪華絢爛なビッグネームたちに交じって臆することなくプレー。クラブ公式ツイッター日本語版（@realmadridjapan）で公開された、ミニゲームで決めた2つのファインゴールの動画が大きな反響を呼んでいる。',
    'フィギュアスケートの全日本強化合宿が１５日、愛知県・豊田市の中京大アイスアリーナで公開された。男子で１８年平昌五輪銀メダリストの宇野昌磨（２１）＝トヨタ自動車＝は今季、メインコーチ不在で戦う意向を示した。「はやく見つけようとは思っていない。今季はおそらく僕一人でやっていく。不安はない。１人でも僕は出来ると思っている」と話した。',
    'レアル・マドリードの日本代表MF久保建英（18）が、インターナショナルチャンピオンズ杯（ICC）全3試合で起用される可能性が高いことが分かった。今季は3部所属のBチームでプレー予定だが、カナダ・モントリオールでのトップチームの合宿で評価は急上昇。14日付のスペイン紙「アス」で1面を飾るなど注目度は高まっており、20日（日本時間21日）のバイエルン・ミュンヘン戦を皮切りに全3戦でテストされる見通しとなった。'
]

docs = [tagger.parse(t).strip().split(' ') for t in texts]
vecs = [model.infer_vector(doc) for doc in docs]

print('Document 1 and 2')
print(cosine_similarity(vecs[0], vecs[1]))
print()
print('Document 2 and 3')
print(cosine_similarity(vecs[1], vecs[2]))
print()
print('Document 1 and 3')
print(cosine_similarity(vecs[0], vecs[2]))