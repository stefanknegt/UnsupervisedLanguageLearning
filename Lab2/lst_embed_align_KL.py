import pickle
import numpy as np
from numpy.linalg import norm

def cos_sim(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))

def kl_div(mu_1, sigma_1, mu_2, sigma_2):
    KL = np.sum(np.log(sigma_2/sigma_1) + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2)/(2* sigma_2 ** 2) - 0.5)
    if KL < 0: #KL is sometime 1e-6 because of epsilon, therefore we set to 0.
        KL = 0
    return 1 / (1 + KL)

def relu(x):
    return np.maximum(x, 0)

def softplus(x):
    return np.log(1 + np.exp(x))

def calc_mu_sigma(word, sentence_embed):
    word_embed = embedding_dict[word]
    concat_embed = np.concatenate((word_embed, sentence_embed), axis=0)

    mu = np.dot(mu_weights, concat_embed) + mu_bias
    sigma = softplus(np.dot(sigma_weights, concat_embed) + sigma_bias)
    return mu, sigma + 1e-6

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
with open('embed_align_embeddings.pickle', 'rb') as handle:
    embedding_dict = pickle.load(handle)
with open('embed_align_mu_sigma_weights.pickle', 'rb') as handle:
    mu_sigma_weights = pickle.load(handle)
    mu_weights, mu_bias, sigma_weights, sigma_bias = mu_sigma_weights[0], mu_sigma_weights[1], mu_sigma_weights[2], mu_sigma_weights[3]

#Main loop to produce out file
output_dict = {}
e_count = 0
for tk, tv in task_dict.items():
    unavailable = False
    sentence = tv["sentence"].split()
    idx = int(tv["pos"])

    centre = tv["replace"]

    #make sentence embedding:
    sentence_embed = np.zeros(100)
    words = 0
    for word in sentence:
        try:
            sentence_embed += embedding_dict[mid]
            words += 1
        except:
            continue
    #normalize
    if words > 0:
        sentence_embed /= words

    try:
        mu_real, sigma_real = calc_mu_sigma(centre.split(".")[0], sentence_embed)
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
                    mu_candidate, sigma_candidate = calc_mu_sigma(c, sentence_embed)
                    score += kl_div(mu_candidate, sigma_candidate, mu_real, sigma_real)
                    count += 1
                except:
                    continue

            if score == 1 or count == 0:
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

with open('output_embed_align.txt', 'w') as f:
    for item in output:
        f.write("%s\n" % item)
