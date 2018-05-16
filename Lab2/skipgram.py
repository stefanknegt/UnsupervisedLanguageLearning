import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
from preprocess import *
import pickle
import time

#Inspiration taken from: https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb

def save_embeddings(W1,W2):
    #Save both the word and context embeddings using pickle
    word_embeddings = W1.data.numpy()
    word_context_embeddings = W2.data.numpy()

    embedding_dict = {}
    context_embedding_dict = {}
    for i in range(0,word_embeddings.shape[1]):
        embedding_dict[idx2word[i]] = word_embeddings[:,i]#.tolist()
        context_embedding_dict[idx2word[i]] = word_context_embeddings[i,:]

    with open('skipgram_embeddings.pickle', 'wb') as handle:
        pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('skipgram_context_embeddings.pickle', 'wb') as handle:
        pickle.dump(context_embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_one_hot(word_idx,vocab_size):
    #Return one hot vector for given center (input) word
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x

def train_skipgram(vocab_size):
    #Initialize parameters and weight matrices
    embedding_dims = 50
    W1 = Variable(torch.randn(embedding_dims, vocab_size).float(), requires_grad=True)
    W2 = Variable(torch.randn(vocab_size, embedding_dims).float(), requires_grad=True)
    num_epochs = 5
    learning_rate = 0.001
    iter = 0
    batch_loss = 0

    for epoch in range(num_epochs):

        epoch_loss = 0
        print("Epoch number:", epoch)
        start = time.time()

        for center, target in pairs:
            iter += 1
            #Get one hot input and target
            x = Variable(make_one_hot(center,vocab_size)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            #Get the activations z1 and z2
            z1 = torch.matmul(W1, x)
            z2 = torch.matmul(W2, z1)

            #Calculate the cross-entropy loss (using lof softmax and negative log-likelihood loss)
            log_softmax = F.log_softmax(z2, dim=0)
            loss = F.nll_loss(log_softmax.view(1,-1), y_true)
            batch_loss += loss

            #Using the loss calculate the gradients
            epoch_loss += batch_loss
            batch_loss.backward()

            #Adjust the weights using the gradients
            W1.data -= learning_rate * W1.grad.data
            W2.data -= learning_rate * W2.grad.data

            #Set gradients to zero
            W1.grad.data.zero_()
            W2.grad.data.zero_()

            batch_loss = 0

            if iter % 1000 == 0:
                print("Iteration ", iter)

        print("Elapsed time for epoch:", time.time()- start)

        if epoch % 2 == 0:
            print(f'Loss at epoch {epoch}: {epoch_loss/len(pairs)}')

    save_embeddings(W1,W2)

#Load the vocabulary
vocab_size,indiced_text,vocab,word2idx,idx2word = make_vocab('hansards/training.en')
"""with open('word2idx.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)
with open('idx2word.pickle', 'rb') as handle:
    idx2word = pickle.load(handle)
with open('indiced_text.pickle', 'rb') as handle:
    indiced_text = pickle.load(handle)
vocab_size = len(word2idx)"""

#Make word context pairs
pairs = word_context_pairs(indiced_text,2,'skipgram')
print("Number of word context pairs:", len(pairs))

#Train the model
train_skipgram(vocab_size)
