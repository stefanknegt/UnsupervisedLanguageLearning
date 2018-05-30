from gensim.models import Word2Vec,KeyedVectors

def train_skipgram():
    with open('hansards/training.en') as f:
        text = f.readlines()
    sentences = [x.strip() for x in text]
    splitted_sentences = [s.split() for s in sentences]
    splitted_sentences = [[w.lower() for w in sentence] for sentence in splitted_sentences]

    print(type(splitted_sentences[0]))
    model = Word2Vec(splitted_sentences, sg=1, size=100, alpha=0.025)
    model.train(splitted_sentences, total_examples=model.corpus_count, epochs=50)
    model.save("skipgram-100d-50e-mincount0.bin")

def test_skipgram(word):
    model = Word2Vec.load('skipgram2.bin')
    print(model)
    word_vectors = model.wv
    word_vectors.save('vectors.txt')
    word_vectors_new = KeyedVectors.load('vectors.txt')
    print(word_vectors_new.most_similar(positive=[word]))


train_skipgram()
#test_skipgram('money')
