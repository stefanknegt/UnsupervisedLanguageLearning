import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
from preprocess import *

#helper functions for one hot and bag of words (unused)
def make_one_hot(word_idx, vocab_size):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x

#Save embeddings in dictionary
def save_embeddings_embed_align(W1):
    word_embeddings = W1.data.numpy()
    embedding_dict = {}
    for i in range(0,word_embeddings.shape[1]):
        embedding_dict[L1_idx2word[i]] = word_embeddings[:,i]#.tolist()
    with open('embed_align_embeddings.pickle', 'wb') as handle:
        pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_embed_align_matrices(mu_weights, sigma_weights, mu_bias, sigma_bias):
    mu_weights = mu_weights.data.numpy()
    sigma_weights = sigma_weights.data.numpy()
    mu_bias = mu_bias.data.numpy()
    sigma_bias = sigma_bias.data.numpy()

    with open('embed_align_mu_sigma_weights.pickle', 'wb') as handle:
        pickle.dump([mu_weights, mu_bias, sigma_weights, sigma_bias], handle, protocol=pickle.HIGHEST_PROTOCOL)

#Helper functions for AER test
def dict_of_sentences(path, word2idx):
    with open('wa/dev.en') as f:
         text = f.readlines()

    sentence_dict = {}
    count = 1
    for sentence in text:
        sentence = sentence.lower()
        sentence = sentence.split()
        sentence_unk = [k if k in word2idx.keys() else 'UNK' for k in sentence]
        sentence_word_idxs = [word2idx[k] for k in sentence_unk]
        sentence_dict[count] = sentence_word_idxs
        count += 1
    return sentence_dict

#AER score function. For every word in L1 we look at the activation of all L2 words - and make the file
def aer_score(embedding_dims, L1_embedding_weights, mu_weights, sigma_weights, mu_bias, sigma_bias, L1_hidden_weights, L1_hidden_bias, L1_out_weights, L1_out_bias, L2_hidden_weights, L2_hidden_bias, L2_out_weights, L2_out_bias):
    align_dict = {}
    L1_align_dict = dict_of_sentences('wa/test.en', L1_word2idx)
    L2_align_dict = dict_of_sentences('wa/test.fr', L2_word2idx)

    for k, L1_sentence in L1_align_dict.items():
        align_dict[k] = []
        L2_sentence = L2_align_dict[k]

        L1_sentence_embed = torch.zeros(embedding_dims).float()
        for L1_word in L1_sentence:
            L1_one_hot = Variable(make_one_hot(L1_word, L1_vocab_size)).float()
            L1_sentence_embed += torch.matmul(L1_embedding_weights, L1_one_hot) / len(L1_sentence)

        for L1_idx in range(len(L1_sentence)):
            #same steps as in the normal network:
            scores = np.zeros(len(L2_sentence))
            L1_word = L1_sentence[L1_idx]

            L1_one_hot = Variable(make_one_hot(L1_word, L1_vocab_size)).float()
            L1_embedding = torch.matmul(L1_embedding_weights, L1_one_hot)
            L1_concat_embed = torch.cat((L1_embedding, L1_sentence_embed), 0)

            mu_z = torch.matmul(mu_weights, L1_concat_embed) + mu_bias
            sigma_z = F.softplus(torch.matmul(sigma_weights, L1_concat_embed.clone()) + sigma_bias)

            epsilon = Variable(torch.randn(embedding_dims)).float()
            sample_z = mu_z + torch.mul(epsilon, sigma_z)

            L1_hidden = F.relu(torch.matmul(L1_hidden_weights, sample_z) + L1_hidden_bias)
            L1_out = (torch.matmul(L1_out_weights, L1_hidden) + L1_out_bias)

            L2_hidden = F.relu(torch.matmul(L2_hidden_weights, sample_z.clone()) + L2_hidden_bias)
            L2_out = (torch.matmul(L2_out_weights, L2_hidden) + L2_out_bias)
            L2_out_softmax = F.softmax(L2_out)

            #for every word, look at the
            for L2_idx in range(len(L2_sentence)):
                L2_word = L2_sentence[L2_idx]
                scores[L2_idx] = L2_out_softmax[L2_word]
            if max(scores) == 0: #if we are not sure, let us append the same index
                align_dict[k].append((L1_idx + 1, L2_idx + 1))
            else:
                align_dict[k].append((L1_idx + 1, np.argmax(scores) + 1))
    #save in the right format
    with open('naacl_files/output_aer.naacl', 'w') as f:
        for k, score_list in align_dict.items():
            for scoring in score_list:
                item = str(k) + ' ' + str(scoring[0]) + ' ' + str(scoring[1]) + ' S'
                f.write("%s\n" % item)
    return

def train_align_embed(L1_indiced_text, L2_indiced_text, L1_vocab_size, L2_vocab_size):
    #Initialize weights and hyper parameters
    embedding_dims = 100
    L1_embedding_weights = Variable(torch.randn(embedding_dims, L1_vocab_size).float(), requires_grad=True)

    mu_weights = Variable(torch.randn(embedding_dims, 2 * embedding_dims).float(), requires_grad=True)
    sigma_weights = Variable(torch.randn(embedding_dims, 2 * embedding_dims).float(), requires_grad=True)
    mu_bias = Variable(torch.randn(embedding_dims).float(), requires_grad=True)
    sigma_bias = Variable(torch.randn(embedding_dims).float(), requires_grad=True)

    L1_hidden_weights = Variable(torch.randn(embedding_dims, embedding_dims).float(), requires_grad=True)
    L1_hidden_bias = Variable(torch.randn(embedding_dims).float(), requires_grad=True)
    L1_out_weights = Variable(torch.randn(L1_vocab_size, embedding_dims).float(), requires_grad=True)
    L1_out_bias = Variable(torch.randn(L1_vocab_size).float(), requires_grad=True)

    L2_hidden_weights = Variable(torch.randn(embedding_dims, embedding_dims).float(), requires_grad=True)
    L2_hidden_bias = Variable(torch.randn(embedding_dims).float(), requires_grad=True)
    L2_out_weights = Variable(torch.randn(L2_vocab_size, embedding_dims).float(), requires_grad=True)
    L2_out_bias = Variable(torch.randn(L2_vocab_size).float(), requires_grad=True)

    num_epochs = 1
    batch_size = 1000
    eps = 1e-6 #for log 0
    alpha = 0
    data_points = 50000

    #define an optimizer for the problem
    optimizer = torch.optim.Adam([L1_embedding_weights, mu_weights, sigma_weights, mu_bias, sigma_bias, L1_hidden_weights, L1_hidden_bias, L1_out_weights, L1_out_bias,
                            L2_hidden_weights, L2_hidden_bias, L2_out_weights, L2_out_bias], lr = 0.001)

    for epoch in range(num_epochs): #for every epoch
        batch_loss = 0 #used for prining
        iter = 0 #used for printing
        for s in range(data_points): #range(len(L1_indiced_text)):
            if alpha < 1:
                alpha += 2 / data_points #we want to be at alpha = 1 at half of datapoints
            iter += 1
            L1_sentence = L1_indiced_text[s]
            L2_sentence = L2_indiced_text[s]

            m = len(L1_sentence) #get lengths
            n = len(L2_sentence)

            #Average sentence embedding:
            L1_sentence_embed = torch.zeros(embedding_dims).float()
            for L1_word in L1_sentence:
                L1_one_hot = Variable(make_one_hot(L1_word, L1_vocab_size)).float()
                L1_sentence_embed += torch.matmul(L1_embedding_weights, L1_one_hot) / m

            for L1_word in L1_sentence:
                #Get word + sentence representation of L1 sentence
                L1_one_hot = Variable(make_one_hot(L1_word, L1_vocab_size)).float()
                L1_embedding = torch.matmul(L1_embedding_weights, L1_one_hot)
                L1_concat_embed = torch.cat((L1_embedding, L1_sentence_embed), 0)

                #Now we make mu and sigma predictions for Z
                mu_z = torch.matmul(mu_weights, L1_concat_embed) + mu_bias
                sigma_z = F.softplus(torch.matmul(sigma_weights, L1_concat_embed.clone()) + sigma_bias)

                #Then we randomly sample a Z
                epsilon = Variable(torch.randn(embedding_dims)).float()
                sample_z = mu_z + torch.mul(epsilon, sigma_z)

                #Then we predict back the L1 word given Z:
                L1_hidden = F.relu(torch.matmul(L1_hidden_weights, sample_z) + L1_hidden_bias)
                L1_out = (torch.matmul(L1_out_weights, L1_hidden) + L1_out_bias)

                #We also predict the L2 word given the Z variable and then we average loss across the sentence
                L2_hidden = F.relu(torch.matmul(L2_hidden_weights, sample_z.clone()) + L2_hidden_bias)
                L2_out = (torch.matmul(L2_out_weights, L2_hidden) + L2_out_bias)

                #So for every word we get the one-hot and calculate the loss.
                L2_loss = 0
                for L2_word in L2_sentence:
                    L2_one_hot = Variable(torch.from_numpy(np.array([L2_word])).long())
                    log_softmax = F.log_softmax(L2_out, dim=0)
                    L2_loss += F.nll_loss(log_softmax.view(1,-1), L2_one_hot)/ m

                #L1 loss for predicting back the word
                L1_one_hot2 = Variable(torch.from_numpy(np.array([L1_word])).long())
                log_softmax = F.log_softmax(L1_out, dim=0)
                L1_loss = F.nll_loss(log_softmax.view(1,-1), L1_one_hot2)

                #Then we also need the KL divergence term, which we calculate as: (q is supposed to be normal distributed N(0,1))
                KL_loss = - 0.5 * torch.sum(1 + torch.log((sigma_z + eps) ** 2) - mu_z ** 2 - sigma_z ** 2)

                loss = L1_loss + L2_loss + alpha * KL_loss #alpha parameter to increase KL term over time
                #Take optimizer step
                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                optimizer.step()

                batch_loss += loss.data.item()

                if (iter != 0 and iter % batch_size == 0 or iter == len(L1_indiced_text)-1):
                    print('Batch loss at epoch, point: ', epoch, s, batch_loss/iter)
                    iter = 0
                    batch_loss = 0
    #Make aer score and save embeddings for subsition task
    aer_score(embedding_dims, L1_embedding_weights, mu_weights, sigma_weights, mu_bias, sigma_bias, L1_hidden_weights, L1_hidden_bias, L1_out_weights, L1_out_bias, L2_hidden_weights, L2_hidden_bias, L2_out_weights, L2_out_bias)
    save_embeddings_embed_align(L1_embedding_weights)
    save_embed_align_matrices(mu_weights, sigma_weights, mu_bias, sigma_bias)
    return

#Open saved dicts:
with open('word2idx_en.pickle', 'rb') as handle:
    L1_word2idx = pickle.load(handle)
with open('idx2word_en.pickle', 'rb') as handle:
    L1_idx2word = pickle.load(handle)
with open('indiced_text_en.pickle', 'rb') as handle:
    L1_indiced_text = pickle.load(handle)
L1_vocab_size = len(L1_word2idx)

with open('word2idx_fr.pickle', 'rb') as handle:
    L2_word2idx = pickle.load(handle)
with open('idx2word_fr.pickle', 'rb') as handle:
    L2_idx2word = pickle.load(handle)
with open('indiced_text_fr.pickle', 'rb') as handle:
    L2_indiced_text = pickle.load(handle)
L2_vocab_size = len(L2_word2idx)

print(len(L2_indiced_text))

#These are used for development
'''
_, L1_indiced_text, L1_vocab, L1_word2idx, L1_idx2word =  make_vocab('wa/dev.en')
_, L2_indiced_text, L2_vocab, L2_word2idx, L2_idx2word =  make_vocab('wa/dev.en')

L1_vocab_size = len(L1_word2idx)
L2_vocab_size = len(L2_word2idx)
'''

#Train model
train_align_embed(L1_indiced_text, L2_indiced_text, L1_vocab_size, L2_vocab_size)
