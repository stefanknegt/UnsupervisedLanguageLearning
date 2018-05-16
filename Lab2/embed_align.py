import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
from preprocess import *

def make_one_hot(word_idx, vocab_size):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x

def make_bow(word_idx_list, vocab_size):
    x = torch.zeros(vocab_size).float()
    for word_idx in word_idx_list:
        x[word_idx] += 1
    return x

def train_align_embed(both_indiced_text, L1_vocab_size, L2_vocab_size):
    embedding_dims = 5
    hidden_dims = 10

    mu_weights_1 = Variable(torch.randn(hidden_dims, L1_vocab_size).float(), requires_grad=True)
    sigma_weights_1 = Variable(torch.randn(hidden_dims, L1_vocab_size).float(), requires_grad=True)
    mu_bias_1 = Variable(torch.randn(hidden_dims).float(), requires_grad=True)
    sigma_bias_1 = Variable(torch.randn(hidden_dims).float(), requires_grad=True)

    mu_weights_2 = Variable(torch.randn(embedding_dims, hidden_dims).float(), requires_grad=True)
    sigma_weights_2 = Variable(torch.randn(embedding_dims, hidden_dims).float(), requires_grad=True)
    mu_bias_2 = Variable(torch.randn(embedding_dims).float(), requires_grad=True)
    sigma_bias_2 = Variable(torch.randn(embedding_dims).float(), requires_grad=True)

    L1_hidden_weights = Variable(torch.randn(hidden_dims, embedding_dims).float(), requires_grad=True)
    L1_hidden_bias = Variable(torch.randn(hidden_dims).float(), requires_grad=True)
    L1_out_weights = Variable(torch.randn(L1_vocab_size, hidden_dims).float(), requires_grad=True)
    L1_out_bias = Variable(torch.randn(L1_vocab_size).float(), requires_grad=True)

    L2_hidden_weights = Variable(torch.randn(hidden_dims, embedding_dims).float(), requires_grad=True)
    L2_hidden_bias = Variable(torch.randn(hidden_dims).float(), requires_grad=True)
    L2_out_weights = Variable(torch.randn(L2_vocab_size, hidden_dims).float(), requires_grad=True)
    L2_out_bias = Variable(torch.randn(L2_vocab_size).float(), requires_grad=True)

    num_epochs = 500
    learning_rate = 0.001
    optimizer = torch.optim.Adam([mu_weights_1, sigma_weights_1, mu_bias_1, sigma_bias_1,mu_weights_2, sigma_weights_2, mu_bias_2, sigma_bias_2, L1_hidden_weights, L1_hidden_bias, L1_out_weights, L1_out_bias,
                            L2_hidden_weights, L2_hidden_bias, L2_out_weights, L2_out_bias], lr = 0.001)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for L1_sentence, L2_sentence in both_indiced_text:
            m = len(L1_sentence)
            n = len(L2_sentence)
            for L1_word in L1_sentence:
                #Get bag of words representation of L1 sentence
                L1_one_hot = Variable(make_one_hot(L1_word, L1_vocab_size)).float()

                #Now we make mu and sigma predictions for Z
                mu_hidden = F.relu(torch.matmul(mu_weights_1, L1_one_hot) + mu_bias_1)
                sigma_hidden = F.relu(torch.matmul(sigma_weights_1, L1_one_hot) + sigma_bias_1)

                mu_z = torch.matmul(mu_weights_2, mu_hidden) + mu_bias_2
                log_sigma_sq_z = torch.matmul(sigma_weights_2, sigma_hidden) + sigma_bias_2

                #Then we randomly sample a Z
                epsilon = Variable(torch.randn(embedding_dims)).float()
                sample_z = mu_z + epsilon * torch.exp(torch.sqrt(log_sigma_sq_z))

                #Then we predict back the L1 word given Z:
                L1_hidden = F.relu(torch.matmul(L1_hidden_weights, sample_z) + L1_hidden_bias)
                L1_out = (torch.matmul(L1_out_weights, L1_hidden) + L1_out_bias)

                #We also predict the L2 word given the Z variable and then we average loss across the sentence
                L2_hidden = F.relu(torch.matmul(L2_hidden_weights, sample_z) + L2_hidden_bias)
                L2_out = (torch.matmul(L2_out_weights, L2_hidden) + L2_out_bias)

                #So for every word we get the one-hot and calculate the loss.
                L2_loss = 0
                for L2_word in L2_sentence:
                    L2_one_hot = Variable(torch.from_numpy(np.array([L2_word])).long())
                    log_softmax = F.log_softmax(L2_out, dim=0)
                    L2_loss = F.nll_loss(log_softmax.view(1,-1), L2_one_hot)/ m

                L1_one_hot2 = Variable(torch.from_numpy(np.array([L1_word])).long())
                log_softmax = F.log_softmax(L1_out, dim=0)
                L1_loss = F.nll_loss(log_softmax.view(1,-1), L1_one_hot2)

                #Then we also need the KL divergence term, which we calculate as: (q is supposed to be normal distributed N(0,1))
                KL_loss = torch.sum(torch.log(1 / torch.exp(torch.sqrt(log_sigma_sq_z))) + (torch.exp(log_sigma_sq_z) + (mu_z - 1)**2) / (2 * torch.exp(log_sigma_sq_z)) - 0.5)

                #Total loss:
                optimizer.zero_grad()
                loss = L1_loss + L2_loss + KL_loss
                loss.backward()

                optimizer.step()

                epoch_loss += loss.data.item()

        if epoch % 10 == 0:
            print(f'Loss at epoch {epoch}: {epoch_loss}')



L1_vocab_size, L1_tokenized_text, L1_indiced_text, L1_vocab, L1_word2idx, L1_idx2word = make_vocab('wa/dev.en')
L2_vocab_size, L2_tokenized_text, L2_indiced_text, L2_vocab, L2_word2idx, L2_idx2word = make_vocab('wa/dev.fr')
both_indiced_text = zip(L1_indiced_text, L2_indiced_text)

train_align_embed(both_indiced_text, L1_vocab_size, L2_vocab_size)
