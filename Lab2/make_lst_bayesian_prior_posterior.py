import pickle
import numpy as np
from numpy.linalg import norm

def cos_sim(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))

def kl_div(mu_1, sigma_1, mu_2, sigma_2):
    KL = np.sum(np.log((sigma_2 + 1e-6)/(sigma_1 + 1e-6)) + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2)/(2* sigma_2 ** 2 + 1e-6) - 0.5)
    return 1 / (1 + KL)

def relu(x):
    return np.maximum(x, 0)

def softplus(x):
    return np.log(1 + np.exp(x))

#Task dict is a dict with as key the sentence id and as value a dict with word to replace, position word and sentence
with open('lst/lst_test.preprocessed') as f:
    task = [line.strip().split('\t') for line in f]
task_dict = {}
for line in task:
    task_dict[line[1]] = {"replace": line[0], "pos": line[2], "sentence":line[3]}

#Read gold standard in as dict with key the word to replace and value a list of candidates
with open('lst/lst.gold.candidates') as f:
    gold = [line.strip() for line in f]
gold_dict = {}
for line in gold:
    gold_dict[line.split('::')[0]] = line.split('::')[1].split(';')

#Load word embeddings (dict)
with open('Models/bayesian_skipgram_mu_sigma_new2.pickle', 'rb') as handle:
    mu_sigma_dict = pickle.load(handle)
with open('Models/bayesian_skipgram_context_embeddings_new2.pickle', 'rb') as handle:
    context_embedding_dict = pickle.load(handle)
with open('Models/bayesian_skipgram_context_mu_sigma_new2.pickle', 'rb') as handle:
    context_weights = pickle.load(handle)
    mu_context_weights, mu_context_bias, sigma_context_weights, sigma_context_bias = context_weights[0], context_weights[1], context_weights[2], context_weights[3]

#Main loop to produce out file
output_dict = {}
e_count = 0
for tk, tv in task_dict.items():
    unavailable = False
    context = tv["sentence"].split()
    idx = int(tv["pos"])
    del context[idx]
    window = context[idx - 2: idx + 2]

    try:
        centre = tv["replace"]
        centre_embed = context_embedding_dict[centre.split(".")[0]]
        concat_embed_sum = np.zeros(len(centre_embed) * 2)
        for word in window:
            try:
                context_embed = context_embedding_dict[word]
                concat_embed_sum += relu(np.concatenate((centre_embed, context_embed), axis=0))
            except :
                continue
        mu_context = np.dot(mu_context_weights, concat_embed_sum) + mu_context_bias
        sigma_context = softplus(np.dot(sigma_context_weights, concat_embed_sum) + sigma_context_bias)
    except Exception as e:
        print(e)
        e_count += 1
        unavailable = True

    candidates = gold_dict[centre]
    scores = []
    if unavailable:
        [scores.append((c, 0)) for c in candidates]
    else:
        for c in candidates:
            c_list = c.split()
            count = 0
            score = 0
            for c in c_list:
                try:
                    mu, sigma = mu_sigma_dict[c]
                    score += cos_sim(mu, mu_context) #kl_div(mu, softplus(sigma), mu_context, sigma_context)
                    count += 1
                except:
                    continue
            if count == 0:
                scores.append((c, 0))
            else:
                scores.append((c, score/count))

    output_dict[(tv["replace"],tk)] = scores

print(e_count)
output = []
for k,v in output_dict.items():
    string = []
    string.append("RANKED")
    string.append((k[0]+" "+k[1]))
    for word in v:
        string.append((word[0]+" "+str(word[1])))
    output.append("\t".join(string))

with open('output_bayesian_prior_post.txt', 'w') as f:
    for item in output:
        f.write("%s\n" % item)
