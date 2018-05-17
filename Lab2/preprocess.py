import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter
import pickle
from random import randint

def make_vocab(filename):
    """
    This function takes a file reads the lines and makes a vocabulary (lower-case)
    It also makes a word2idx and idx2word dictionary and transforms the sentences to indices only
    """

    print("Make vocab")
    translator = str.maketrans('', '', string.punctuation)

    with open(filename) as f:
        text = f.readlines()

    text = [x.strip() for x in text]

    text = text

    no_stopwords_text = []

    for sentence in text:
        sentence = sentence.lower()
        sentence = sentence.translate(translator) #remove punctation CHECK FOR + - / * : etc
        no_stopwords_text.append(" ".join([word for word in sentence.split() if word not in stopwords.words('english')])) #remove stopwords

    text = no_stopwords_text

    tokenized_text = [x.split() for x in text]

    flat_list = [item for sublist in tokenized_text for item in sublist]
    counter_text = Counter(flat_list)

    vocab = []
    indiced_text = []
    for sentence in tokenized_text:
        for word in sentence:
            if counter_text[word] > 2 and word not in vocab:
                vocab.append(word)

    word2idx = {w: idx for (idx, w) in enumerate(vocab)}
    idx2word = {idx: w for (idx, w) in enumerate(vocab)}

    length = len(word2idx)

    word2idx["UNK"] = length
    idx2word[length] = "UNK"

    print("Vocab size is: ", len(word2idx))

    for sentence in tokenized_text:
        indiced_text.append([word2idx[word]if word in word2idx.keys() else word2idx["UNK"] for word in sentence])

    vocab_size = len(vocab)

    with open('word2idx_fr.pickle', 'wb') as handle:
        pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('idx2word_fr.pickle', 'wb') as handle:
        pickle.dump(idx2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('indiced_text_fr.pickle', 'wb') as handle:
        pickle.dump(indiced_text, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return vocab_size,indiced_text,vocab,word2idx,idx2word

def word_context_pairs(indiced_text,window_size,type):
    """
    This function takes the text (indices) and a window size
    Then it determines the pairs (tuples) for the simple skipgram model
    For the advanced skipgram model it also returns negative examples
    For the Bayesian model (other) it returns a dictionary
    with center word as key and a list of context words as values
    """
    print("Make word context pairs")
    #Simple skipgram
    if type == 'skipgram':
        pairs = []
        for sentence in indiced_text:
            for central_word_pos in range(0,len(sentence)):
                for window in range(-window_size,window_size+1):
                    context_word_pos = central_word_pos + window
                    if (context_word_pos <= 0 or context_word_pos >= len(sentence) or context_word_pos == central_word_pos):
                        continue
                    else:
                        pairs.append((sentence[central_word_pos],sentence[context_word_pos]))

        return np.array(pairs)

    #Skipgram with negative sampling
    elif type == 'skipgram_ns':
        pairs = []

        for sentence in indiced_text:
            for central_word_pos in range(0,len(sentence)):
                pos_word_idxs = []
                neg_word_idxs = []

                for window in range(-window_size,window_size+1):
                    context_word_pos = central_word_pos + window
                    if (context_word_pos <= 0 or context_word_pos >= len(sentence) or context_word_pos == central_word_pos):
                        continue
                    else:
                        pos_word_idxs.append(sentence[context_word_pos])

                if len(pos_word_idxs) > 1:
                    for word_idx in sentence:
                        if word_idx not in pos_word_idxs:
                            neg_word_idxs.append(word_idx)
                if len(neg_word_idxs) > 1:

                    for i in range(0,len(pos_word_idxs)):
                        neg_word = neg_word_idxs[randint(0,len(neg_word_idxs)-1)]
                        pos_word = pos_word_idxs[i]
                        if neg_word != pos_word:
                            pairs.append([sentence[central_word_pos],pos_word,neg_word])

        return np.array(pairs)
    #Bayesian skipgram
    elif type == 'other':
        pairs = {}

        for sentence in indiced_text:
            for central_word_pos in range(0,len(sentence)):
                pairs[sentence[central_word_pos]] = []
                for window in range(-window_size,window_size+1):
                    context_word_pos = central_word_pos + window
                    if (context_word_pos <= 0 or context_word_pos >= len(sentence) or context_word_pos == central_word_pos):
                        continue
                    else:
                        pairs[sentence[central_word_pos]].append(sentence[context_word_pos])
        return pairs
