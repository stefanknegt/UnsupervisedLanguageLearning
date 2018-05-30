# Unsupervised Language Learning
Code for the ULL course at UvA 17/18

## Instructions for running the code of Lab1
There is only one ipython notebook which can be used to replicate all the results.

## Instructions for running the code of Lab2
### Training the models
There are two skipgram models. One that uses negative sampling and batches which is called skipgram_with_negative_sampling_and_batches.py and one without both negative sampling and batches called skipgram.py. If you download the entire folder you can simply run them by running python3 filename.py.  The file bayesian_skipgram.py trains the bayesian skipgram model and the file embed_align.py trains the embed align model, there are no further instructions needed for running these models. One can adjust the number of epochs in the code and the weights are automatically stored after training.

### Evaluating the models
As explained in the report we use different methods to construct our output to the lexical substitution task (LST). make_lst_skipgram.py can be used to for the skipgram model, make_lst_bayesian_##.py can be used for the bayesian skipgram model and depending on the file name the different methods are applied. Finally, the file lst_embed_align_KL.py can be used to construct the output for the LST taks using the embed align model. Simply renaming the output file to lst.out and moving it into the /lst folder is enough to run the provided script that determines the generalized average precision.

### Extra files
Preprocess.py is used by all models to construct the vocabulary and training pairs. The file find_most_similar_words.py uses cosine similarity to find similar words using the trained embeddings and was solely used to test the obtained embeddings.

## Instructions for running the code of Lab3
### Training the skipgram model
The skipgram model can be trained by running the train_skipgram.py file. This generates a .bin model file which can be used in combination with the Gensim package.
