from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load('jawiki_doc2vec_dmpv200d.model')
print(model.infer_vector(['私', 'は', '猫人', 'だ']).shape)