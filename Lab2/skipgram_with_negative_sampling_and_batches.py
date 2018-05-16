import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import torch.nn as nn
from preprocess import *
import pickle
import time
from random import shuffle
import torch.optim as optim



class Skipgram(nn.Module):

    #Inspiration taken from: https://github.com/fanglanting/skip-gram-pytorch

    def __init__(self,embedding_dim,vocab_size,batch_size):
        super(Skipgram, self).__init__()
        self.emb_dim = embedding_dim
        self.vocab_size = vocab_size

        #Set hyperparamters
        self.batch_size = batch_size
        self.window_size = 2
        self.num_epochs = 50
        self.learning_rate = 0.01

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.skip_gram_model.cuda()

        self.W1 = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.W2 = nn.Embedding(vocab_size, embedding_dim, sparse=True)

        #Initialize the embedding
        initrange = 0.5 / self.emb_dim
        self.W1.weight.data.uniform_(-initrange, initrange)
        self.W2.weight.data.uniform_(-0, 0)

    def forward_pass(self,pos_c,pos_n,neg_n,batch_size):
        #Make a forward pass and determine the loss
        
        embed_u = self.W1(pos_c)
        embed_v = self.W2(pos_n)

        pos_score  = torch.mul(embed_u, embed_v)
        pos_score = torch.sum(pos_score, dim=1)
        log_sigmid_pos_score = F.logsigmoid(pos_score).squeeze()

        neg_embed_v = self.W2(neg_n)

        neg_score = torch.mul(neg_embed_v, embed_u).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        log_sigmid_neg_score = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_sigmid_pos_score + log_sigmid_neg_score

        return -1 * loss.sum() / batch_size

def save_embeddings(W1,W2,i2w):
    #Store the word and context embeddings using pickle
    word_embeddings = {}
    word_context_embeddings = {}

    for word in i2w:
        w_i = W1.weight.data[word].numpy()
        c_i = W2.weight.data[word].numpy()
        word_embeddings[idx2word[word]] = w_i
        word_context_embeddings[idx2word[word]] = c_i

    with open('skipgram_embeddings.pickle', 'wb') as handle:
        pickle.dump(word_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('skipgram_context_embeddings.pickle', 'wb') as handle:
        pickle.dump(word_context_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_batch(pairs,batch_size,iteration):
    #This function returns a batch with batch_size centers,
    #positive neighbours and negative neighbours

    centers = []
    neighbours = []
    neg_neighbours = []

    #Here we make the batch smaller in case we are at the end of the epoch
    if (iteration + batch_size) > len(pairs)-1:
        max_iter = len(pairs)
        return_iteration = 0
    else:
        max_iter = iteration + batch_size
        return_iteration = max_iter

    for i in range(iteration,max_iter):
        centers.append(pairs[i][0])
        neighbours.append(pairs[i][1])
        neg_neighbours.append(pairs[i][2])

    if cuda_bool == False:
        centers = Variable(torch.LongTensor(centers))
        neighbours = Variable(torch.LongTensor(neighbours))
        neg_neighbours = Variable(torch.LongTensor(neg_neighbours))

    elif cuda_bool == True:
        centers = Variable(torch.LongTensor(centers)).cuda()
        neighbours = Variable(torch.LongTensor(neighbours)).cuda()
        neg_neighbours = Variable(torch.LongTensor(neg_neighbours)).cuda()

    return centers,neighbours,neg_neighbours,return_iteration

#Load the vocabulary
#vocab_size,indiced_text,vocab,word2idx,idx2word = make_vocab('Data/hansards/training.fr')
with open('word2idx_en.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)
with open('idx2word_en.pickle', 'rb') as handle:
    idx2word = pickle.load(handle)
with open('indiced_text_en.pickle', 'rb') as handle:
    indiced_text = pickle.load(handle)

vocab_size = len(word2idx)

#Make word pairs, these are triples over center, pos context, neg context
pairs = word_context_pairs(indiced_text,2,'skipgram_ns')
total_pairs = len(pairs)
shuffle(pairs)

print("Begin training with vocab size " + str(vocab_size) + " and " + str(total_pairs) + " number of pairs")

batch_size = 50
skipgram_model = Skipgram(100,vocab_size,batch_size)
optimizer = optim.SGD(skipgram_model.parameters(),lr=skipgram_model.learning_rate)
cuda_bool = skipgram_model.use_cuda
iter = 0
batch_loss = 0
epoch = 0
epoch_loss = 0

for epoch in range(0,skipgram_model.num_epochs):

    print("Starting epoch number:", epoch)
    start = time.time()
    epoch_done = False

    while epoch_done == False:
        #Get a new batch
        centers,neighbours,neg_neighbours,iter = get_batch(pairs,batch_size,iter)

        #Determine the loss
        loss = skipgram_model.forward_pass(centers,neighbours,neg_neighbours,batch_size)
        epoch_loss += loss.item()

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter == 0:
            print("Time elapsed for epoch: ", time.time() - start)
            print("Loss for epoch " + str(epoch) + " this epoch: ", epoch_loss)
            epoch_loss = 0
            epoch_done = True

save_embeddings(skipgram_model.W1,skipgram_model.W2,idx2word)
