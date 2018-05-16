import pickle
import numpy as np
from numpy.linalg import norm

def cos_sim(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))

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
with open('Models/skipgram_embeddings_100.pickle', 'rb') as handle:
    embeddings = pickle.load(handle)

with open('Models/skipgram_context_embeddings_100.pickle', 'rb') as handle:
    context_embeddings = pickle.load(handle)

#Main loop to produce out file
output_dict = {}
for tk,tv in task_dict.items():

    word_to_replace = tv["replace"]

    context1 = "NA"
    context2 = "NA"
    context3 = "NA"
    context4 = "NA"

    sentence = tv["sentence"].split()

    if type(tv["pos"] != int):
        tv["pos"] = int(tv["pos"])
    if tv["pos"] - 1 >= 0:
        context1 = sentence[tv["pos"]-1]
    if tv["pos"] + 1 < len(sentence):
        context2 = sentence[tv["pos"]+1]
    if tv["pos"] - 2 >= 0:
        context3 = sentence[tv["pos"]-2]
    if tv["pos"] + 2 < len(sentence):
        context4 = sentence[tv["pos"]+2]


    candidates = gold_dict[word_to_replace]
    scores = []
    for c in candidates:
        c_list = c.split()
        if len(c_list) == 1:
            count = 0
            try:
                cs = cos_sim(embeddings[word_to_replace.split(".")[0]],embeddings[c_list[0]])
            except:
                scores.append((c,0.0))
            try:
                if context1 != "NA":
                    count += 1
                    c1 = cos_sim(embeddings[context1],context_embeddings[c_list[0]])
                    cs = cs + c1
            except:
                next
            try:
                if context2 != "NA":
                    c2 = cos_sim(embeddings[context2],context_embeddings[c_list[0]])
                    cs = (cs + c2)
                    count +=1
            except:
                next
            try:
                if context3 != "NA":
                    count += 1
                    c3 = cos_sim(embeddings[context3],context_embeddings[c_list[0]])
                    cs = cs + c3
            except:
                next
            try:
                if context4 != "NA":
                    c4 = cos_sim(embeddings[context2],context_embeddings[c_list[0]])
                    cs = (cs + c4)
                    count +=1
            except:
                next

            if count == 0:
                scores.append((c,cs))
            elif count == 1:
                scores.append((c,(cs/2)))
            elif count == 2:
                scores.append((c,(cs/3)))
            else:
                scores.append((c,(cs/4)))
        else:
            c_embed = np.zeros(100,)
            c_context_embed = np.zeros(100,)

            for word in c_list:
                try:
                    c_embed += embeddings[word]
                    c_context_embed += context_embeddings[word]
                except:
                    next

            c_embed = c_embed/len(c_list)
            c_context_embed = c_context_embed/len(c_list)

            count = 0

            try:
                cs = cos_sim(embeddings[word_to_replace.split(".")[0]],c_embed)
            except:
                scores.append((c,0.0))
            try:
                if context1 != "NA":
                    count += 1
                    c1 = cos_sim(embeddings[context1],c_context_embed)
                    cs = cs + c1
            except:
                next
            try:
                if context2 != "NA":
                    c2 = cos_sim(embeddings[context2],c_context_embed)
                    c2 = (cs + c2)
                    count += 1
            except:
                next
            try:
                if context3 != "NA":
                    count += 1
                    c3 = cos_sim(embeddings[context3],c_context_embed)
                    cs = cs + c3
            except:
                next
            try:
                if context4 != "NA":
                    c4 = cos_sim(embeddings[context2],c_context_embed)
                    cs = (cs + c4)
                    count +=1
            except:
                next

            if count == 0:
                scores.append((c,cs))
            elif count == 1:
                scores.append((c,(cs/2)))
            elif count == 2:
                scores.append((c,(cs/3)))
            else:
                scores.append((c,(cs/4)))

    output_dict[(tv["replace"],tk)] = scores

output = []
for k,v in output_dict.items():
    string = []
    string.append("RANKED")
    string.append((k[0]+" "+k[1]))
    for word in v:
        string.append((word[0]+" "+str(word[1])))
    output.append("\t".join(string))

with open('output.txt', 'w') as f:
    for item in output:
        f.write("%s\n" % item)
