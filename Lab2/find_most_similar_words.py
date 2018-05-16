import pickle
import numpy as np
from numpy.linalg import norm
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

#This file is used to test word embeddings and find the most similar words

with open('skipgram_embeddings30.pickle', 'rb') as handle:
    embeddings = pickle.load(handle)

sims = {}
for key,value in embeddings.items():
    sims[key] = np.dot(embeddings['slowly'], value)/(norm(embeddings['slowly'])*norm(value)) #cosine_similarity([embeddings['corporations']],[value])

print(sorted(sims.items(), key=lambda x: x[1]))
