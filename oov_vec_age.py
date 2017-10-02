import os
import sys
import string
import numpy
import cPickle
import numpy as np
import nltk

import pdb


print "loading GloVe..."
fname = '/home/hantek/datasets/glove/glove.840B.300d.dict.pkl'
if os.path.isfile(fname):
    w1 = cPickle.load(open(fname, 'rb'))
else:
    w1 = {}
    vec = open('/home/hantek/datasets/glove/glove.840B.300d.txt', 'r')
    for line in vec.readlines():
        line=line.split(' ')
        w1[line[0]] = np.asarray([float(x) for x in line[1:]]).astype('float32')
    vec.close()
    save_file = open(fname, 'wb')
    cPickle.dump(w1, save_file)
    save_file.close()


data_choice = sys.argv[1]
if data_choice == 'age1':
    f1 = open('/home/hantek/datasets/NLC_data/age1/age1_train', 'r')
    f2 = open('/home/hantek/datasets/NLC_data/age1/age1_valid', 'r')
    f3 = open('/home/hantek/datasets/NLC_data/age1/age1_test', 'r')
elif data_choice == 'age2':
    f1 = open('/home/hantek/datasets/NLC_data/age2/age2_train', 'r')
    f2 = open('/home/hantek/datasets/NLC_data/age2/age2_valid', 'r')
    f3 = open('/home/hantek/datasets/NLC_data/age2/age2_test', 'r')
    classname = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
else:
    raise ValueError("command line argument has to be either 'age1' or 'age2'.")

f = [f1, f2, f3]

print "processing dataset, 3 dots to punch: ",
sys.stdout.flush()
w2 = {}
w_referred = {0: 0}  # reserve 0 for future padding
vocab_count = 1  # 0 is reserved for future padding
train_dev_test = []
for file in f:
    print ".",
    sys.stdout.flush()
    pairs = []
    for line in file.readlines():
        line=line.decode('utf-8').split('\t')
        s1 = nltk.word_tokenize(line[1])
        s1[0]=s1[0].lower()

        rate_score = line[0]

        s1_words = []
        for word in s1:
            # strip some possible weird punctuations
            word = word.strip(string.punctuation)
            if not w_referred.has_key(word):
                w_referred[word] = vocab_count
                vocab_count += 1
            s1_words.append(w_referred[word])
            if not w1.has_key(word):
                if not w2.has_key(word):
                    w2[word]=[]
                # find the WE for its surounding words
                for neighbor in s1:
                    if w1.has_key(neighbor):
                        w2[word].append(w1[neighbor])

        pairs.append((numpy.asarray(s1_words).astype('int32'),
                      numpy.asarray(rate_score).astype('int32')))

    train_dev_test.append(pairs)
    file.close()

print "\nNumber of UNK tokens:   " + str(len(w2.keys()))
print "          known tokens: " + str(len(w_referred.keys()))
#                       age1           age2
# UNK                                43256
# Known words                       120739
# training set size:                 68485
# dev and test size:                  4000

print "\naugmenting word embedding vocabulary..."
# this block is causing memory error in a 8G computer. Using alternatives.
# all_sentences = [w2[x] for x in w2.iterkeys()]
# all_words = [item for sublist in all_sentences for item in sublist]
# mean_words = np.mean(all_words)
# mean_words_std = np.std(all_words)
mean_words = np.zeros((300,))
mean_words_std = 1e-1

npy_rng = np.random.RandomState(123)
for k in w2.iterkeys():
    if len(w2[k]) != 0:
        w2[k] = sum(w2[k]) / len(w2[k])  # mean of all surounding words
    else:
        # len(w2[k]) == 0 cases: ['cantunderstans', 'motocyckes', 'arefun']
        # I hate those silly guys...
        w2[k] = mean_words + npy_rng.randn(mean_words.shape[0]) * \
                             mean_words_std * 0.1

w2.update(w1)

print "generating weight values..."
# reverse w_referred's key-value;
inv_w_referred = {v: k for k, v in w_referred.items()}

# number   --inv_w_referred-->   word   --w2-->   embedding
ordered_word_embedding = [numpy.zeros((1, 300), dtype='float32'), ] + \
    [w2[inv_w_referred[n]].reshape(1, -1) for n in range(1, len(inv_w_referred))]

# to get the matrix
weight = numpy.concatenate(ordered_word_embedding, axis=0)


print "dumping converted datasets..."
if data_choice == 'age1':
    save_file = open('/home/hantek/datasets/NLC_data/age1/age1.pkl', 'wb')
elif data_choice == 'age2':
    save_file = open('/home/hantek/datasets/NLC_data/age2/age2.pkl', 'wb')

cPickle.dump("dict: truth values and their corresponding class name\n"
             "the whole dataset, in list of list of tuples: list of train/valid/test set -> "
                "list of sentence pairs -> tuple with structure:"
                "(hypothesis, premise, truth class), all entries in numbers\n"
             "numpy.ndarray: a matrix with all referred words' embedding in its rows,"
                "embeddings are ordered by their corresponding word numbers.\n"
             "dict: the augmented GloVe word embedding. contains all possible tokens in SNLI."
                "All initial GloVe entries are included.\n"
             "dict w_referred: word to their corresponding number\n"
             "inverse of w_referred, number to words\n",
             save_file)
cPickle.dump(classname, save_file)
cPickle.dump(train_dev_test, save_file)
cPickle.dump(weight, save_file)
fake_w2 = None; cPickle.dump(fake_w2, save_file)
# cPickle.dump(w2, save_file)  # this is a huge dictionary, delete it if you don't need.
cPickle.dump(w_referred, save_file)
cPickle.dump(inv_w_referred, save_file)
save_file.close()
print "Done."


# check:
def reconstruct_sentence(sent_nums):
    sent_words = [inv_w_referred[n] for n in sent_nums]
    return sent_words

def check_word_embed(sent_nums):
    sent_words = reconstruct_sentence(sent_nums)

    word_embeds_from_nums = [weight[n] for n in sent_nums]
    word_embeds_from_words = [w2[n] for n in sent_words]

    error = 0.
    for i, j in zip(word_embeds_from_nums, word_embeds_from_words):
        error += numpy.sum(i-j)
    
    if error == 0.:
        return True
    else:
        return False
