import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
from preprocess import *
import pickle

def save_embeddings_bayesian(W1):
    word_embeddings = W1.data.numpy()

    embedding_dict = {}
    for i in range(0,word_embeddings.shape[1]):
        embedding_dict[idx2word[i]] = word_embeddings[:,i]#.tolist()

    with open('bayesian_skipgram_embeddings.pickle', 'wb') as handle:
        pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_mu_sigma_bayesian(mu_weights, sigma_weights):
    mu_embeddings = mu_weights.data.numpy()
    sigma_embeddings = sigma_weights.data.numpy()

    mu_sigma_dict = {}
    for i in range(0, mu_embeddings.shape[1]):
        mu_sigma_dict[idx2word[i]] = (mu_embeddings[:,i], sigma_embeddings[:,i])

    with open('bayesian_skipgram_mu_sigma.pickle', 'wb') as handle:
        pickle.dump(mu_sigma_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_context_matrices(embed_weights, mu_weights, sigma_weights, mu_bias, sigma_bias):
    word_embeddings = embed_weights.data.numpy()
    mu_context_weights = mu_weights.data.numpy()
    sigma_context_weights = sigma_weights.data.numpy()
    mu_context_bias = mu_bias.data.numpy()
    sigma_context_bias = sigma_bias.data.numpy()

    embedding_dict = {}
    for i in range(0,word_embeddings.shape[1]):
        embedding_dict[idx2word[i]] = word_embeddings[:,i]

    with open('bayesian_skipgram_context_embeddings.pickle', 'wb') as handle:
        pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('bayesian_skipgram_context_mu_sigma.pickle', 'wb') as handle:
        pickle.dump([mu_context_weights, mu_context_bias, sigma_context_weights, sigma_context_bias], handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_one_hot(word_idx,vocab_size):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x

def train_bayesian_skipgram(vocab_size):
    embedding_dims = 100

    prior_mu_weights = Variable(torch.randn(embedding_dims, vocab_size).float(), requires_grad=True)
    prior_sigma_weights = Variable(torch.randn(embedding_dims, vocab_size).float(), requires_grad=True)

    embed_weights = Variable(torch.randn(embedding_dims, vocab_size).float(), requires_grad=True)
    mu_weights = Variable(torch.randn(embedding_dims, 2*embedding_dims).float(), requires_grad=True)
    sigma_weights = Variable(torch.randn(embedding_dims, 2*embedding_dims).float(), requires_grad=True)
    predict_weights = Variable(torch.randn(vocab_size, embedding_dims).float(), requires_grad=True)

    mu_bias = Variable(torch.randn(embedding_dims).float(), requires_grad=True)
    sigma_bias = Variable(torch.randn(embedding_dims).float(), requires_grad=True)

    num_epochs = 5
    eps = 1e-6
    learning_rate = 0.001
    iter = 0
    batch_size = 1000
    optimizer = torch.optim.Adam([prior_mu_weights, prior_sigma_weights, embed_weights, mu_weights, sigma_weights, predict_weights, mu_bias, sigma_bias], lr = 0.001)

    for epoch in range(num_epochs):
        epoch_loss = 0
        iter = 0
        for centre, context in windows.items():
            iter += 1

            #first we create the mu and sigma of the sum of the concatenated centre+context
            centre_onehot = Variable(make_one_hot(centre,vocab_size)).float()
            centre_embed = torch.matmul(embed_weights, centre_onehot)

            concat_embed_sum = torch.zeros(2*embedding_dims).float()
            for word in context:
                context_onehot = Variable(make_one_hot(word,vocab_size)).float()
                context_embed = torch.matmul(embed_weights, context_onehot)

                concat_embed_sum += F.relu(torch.cat((centre_embed, context_embed), 0))

            mu_z = torch.matmul(mu_weights, concat_embed_sum) + mu_bias
            sigma_z = F.softplus(torch.matmul(sigma_weights, concat_embed_sum) + sigma_bias)

            #then we sample a Gaussian for this mu and sigma and make a prediction over words.
            sample_z = mu_z + sigma_z * Variable(torch.randn(embedding_dims))
            out = torch.matmul(predict_weights, sample_z)
            softmax_out = F.log_softmax(out)

            #then we calculate the entropy loss for the context words
            entropy_loss = 0
            for word in context:
                y_true = Variable(torch.from_numpy(np.array([word])).long())
                entropy_loss += F.nll_loss(softmax_out.view(1,-1), y_true)

            #Now we want the KL divergence between the word embedding and the generated embedding
            mu_prior = torch.matmul(prior_mu_weights, centre_onehot)
            sigma_prior = F.softplus(torch.matmul(prior_sigma_weights, centre_onehot))

            KL_loss = torch.sum(torch.log((sigma_prior + eps)/(sigma_z + eps)) + (sigma_prior**2 + (mu_z - mu_prior)**2) / (2 * sigma_prior**2 + eps) - 0.5)
            loss = (entropy_loss + KL_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()

            if (iter != 0 and iter % batch_size == 0 or iter == len(windows)-1):
                print('Loss at epoch, point: ', epoch, iter, epoch_loss/iter)

    save_embeddings_bayesian(prior_mu_weights)
    save_mu_sigma_bayesian(prior_mu_weights, prior_sigma_weights)
    save_context_matrices(embed_weights, mu_weights, sigma_weights, mu_bias, sigma_bias)

#vocab_size,indiced_text,vocab,word2idx,idx2word = make_vocab('hansards/training.fr')
with open('word2idx_en.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)
with open('idx2word_en.pickle', 'rb') as handle:
    idx2word = pickle.load(handle)
with open('indiced_text_en.pickle', 'rb') as handle:
    indiced_text = pickle.load(handle)
vocab_size = len(word2idx)

windows = word_context_pairs(indiced_text,2,'other')
train_bayesian_skipgram(vocab_size)
