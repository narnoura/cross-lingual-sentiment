# Cross-lingual Sentiment LSTM implementation using Keras
# Can be run cross-lingually or monolingually 
# (i.e, it can be trained and tested on the same language or on different languages)
# Untargeted sentiment (not specific towards topics)

# Importing python libraries
import sys, os, codecs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import collections
from collections import OrderedDict
import pickle
import codecs
import argparse
import csv

# Importing the Keras libraries and packages
from keras.models import Sequential, load_model, Model 
from keras.layers import Dense, Input, Activation
from keras.layers import LSTM, Bidirectional, GlobalAveragePooling1D
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils import cosine_similarity, load_embeddings, read_bilingual_dict, read_sentiment_dict, read_clusters

EMBED_FILE = "/proj/nlp/users/noura/Low-Resource/multivec-master/models-LDC/ar-100-5/en-ar.both.txt"
#EMBED_FILE = "/proj/nlp/senti_transfer/xlingual/mono_embedding/en"
OUT_DIR = "/proj/nlp/users/noura/deep-learning/Experiments/Sentiment/simpleLSTM"
THIS_DIR = "/proj/nlp/users/noura/deep-learning/Experiments/Sentiment"
SENT_LEX_PATH = "/proj/nlp/users/noura/English/lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"
OTHER_EMOJS = [":)", ":(", ":-)", ":-(", ":|", ":O"]
COMMON_WORDS = []


# keep it alphabetical like get_dummies
idx2sentiment = {0: "negative", 1: "neutral", 2: "positive"}
sentiment2idx = {"negative": 0, "neutral": 1, "positive": 2}
fivecls2sent = {1:"negative",2:"negative",3:"neutral",4:"positive",5:"positive"}
# for Chinese hotel review ratings
idx2fivecls = {0:"1",1:"2",2:"3",3:"4",4:"5"}



def text_to_sequence(train_line,
                     word2idx,
                     lexicalize=0,
                     bilingual_dict=None,
                     lower = True,
                     filter_tokens = None):
    '''
    Given a sentence and vocabulary index mapping, tokenizes the sentence into words and maps each word to its index
    If lexicalize is true, the function will attempt to find a translation of the word in dict (assumed to be to the target language),
    and if it finds the translation, it will use the mapping of the translation instead.
    TODO return how many words we find
    '''
    seq = []
    translated = 0
    words = train_line.strip().split(' ')
    for w in words:
        if lower:
            w = w.lower()
        if lexicalize and w in bilingual_dict and bilingual_dict[w] in word2idx and (filter_tokens is None or w in filter_tokens):
            seq.append(word2idx[bilingual_dict[w]])
            translated +=1
        elif w in word2idx and (filter_tokens is None or w in filter_tokens):
            seq.append(word2idx[w])
        else:
            seq.append(0)
    return seq, translated, len(words)

def sentiment_similarity(train_line, word_vectors, sentiment_vectors, method='cosine', lower = True):
    '''
    Returns a sentiment matrix for consisting of similarity of the word to pretrained sentiment vectors
    '''
    seq = []
    words = train_line.strip().split(' ')
    for w in words:
        if lower:
            w = w.lower()
        if w in word_vectors:
            sim_vec = np.array([cosine_similarity(sentiment_vectors[label], word_vectors[w]) for label in sentiment_vectors])
            seq.append(sim_vec)
        else:
            seq.append(np.array([0.0, 0.0, 0.0]))
    return seq

def write_to_file(X_test, y_pred, out_file, encoding='utf-8'):
    of = codecs.open(out_file,'w',encoding)
    for i in range(len(X_test)):
        of.write(X_test[i] + '\t' + y_pred[i] + '\n')
        
def write_to_file_probs(X_test, y_pred, out_file, encoding='utf-8'):
    of = codecs.open(out_file,'w',encoding)
    for i in range(len(X_test)):
        of.write(X_test[i] + '\t' + y_pred[i][0] + '\t' + str(y_pred[i][1]) + '\n')

def get_label_idx(one_hot_data):
    label_idxs = []
    for row in one_hot_data:
        for j in range(len(row)):
            if row[j]==1:
                label_idxs.append(j)
                break
    return label_idxs


parser = argparse.ArgumentParser(description='Keras implementation of cross-lingual sentiment LSTM')
parser.add_argument('--train', dest = 'train_file', help = 'path to training file', required=False, default=None)
parser.add_argument('--test', dest = 'test_file', help = 'path to test file', required=True, default=None)
parser.add_argument('--labels', dest = 'has_labels', help = 'true if test file has labels', required=False, type = int, default = 1)
parser.add_argument('--embed', dest = 'embed_file', help = 'path to pretrained embedding file', required=False, default = EMBED_FILE)
parser.add_argument('--sentiment_embed', dest = 'sentiment_embed_file', help = 'pretrained sentiment vectors for sentiment labels', required=False, default = None)
parser.add_argument('--embed_dim', dest = 'embed_dim', help = 'embedding dimension', required=False, type = int, default = 100)
parser.add_argument('--updatable_embed', dest = 'updatable_embed', help = 'if true, add updatable embedding layer', required=False, type = int, default = 0)
parser.add_argument('--update_pretrained', dest = 'update_pretrained', help = 'if true, update pretrained embedding weights', required=False, type = int, default = 0)
parser.add_argument('--cluster_file', dest = 'cluster_file', help = 'file with cluster words and ids for cluster embeddings', required=False, default = None)
parser.add_argument('--cluster_dim', dest = 'cluster_dim', help = 'dimension for cluster embeddings', required=False, default = 50)
parser.add_argument('--hidden_units', dest = 'hidden_units', help = 'hidden unit dimension', required=False, type = int, default = 100)
parser.add_argument('--pretrain', dest = 'pretrain', help = 'true if using pretrained embeddings', required=False, type = int, default = 1)
parser.add_argument('--outdir', dest = 'out_dir', help = 'output directory', required=False, default = OUT_DIR)
parser.add_argument('--modelfile', dest = 'model_file', help = 'model to load from existing file', required=False, default = "model.crossling.simple")
parser.add_argument('--paramfile', dest = 'param_file', help = 'param file to load from existing file', required=False, default = "model.crossling.simple.params")
parser.add_argument('--outfile', dest = 'out_file', help = 'output file for writing sentiment predictions', required=False, default = "model.crossling.simple.out")
parser.add_argument('--batch_size', dest = 'batch_size', required = False, default = 32, type = int)
parser.add_argument('--epochs', dest = 'epochs', required = False, default = 5, type = int)
parser.add_argument('--bidirectional', dest = 'bidirectional', required = False, default = 1, type = int)
parser.add_argument('--lexicalize', dest = 'lexicalize', help = 'if true, lexicalize foreign language data with the help of a dictionary', type = int, required=False, default = 0)
parser.add_argument('--dict', dest = 'bilingual_dict', help = 'path to bilingual dictionary for lexicalization', required=False, default = None)
parser.add_argument('--targets', dest = 'sentiment_targets', help = 'if true, input data is annotated with sentiment targets (inp_token//T)',
                    required = False, default = 0, type = int)
parser.add_argument('--target_embed', dest = 'target_embed', help = 'if true, use an updatable target embedding layer for target words', required = False, default = 0, type = int)
parser.add_argument('--encoding', dest = 'encoding', help = 'file encoding, specify latin-1 for some languages, otherwise default', required=False, default = "utf8")
parser.add_argument('--label_encoding', dest = 'label_encoding', help = 'specify 5-cls if expecting 1-5 scale data for 3-scale training', required=False, default = "3-cls")
parser.add_argument('--output_units', dest = 'output_units', help = 'number of output units in final dense layer', required=False, type = int, default = 3)

args = parser.parse_args()

train = 0
predict = 0

# For sentiment embeddings
idx2sentvectors = OrderedDict()
sent_idx = 1

# Importing the datasets
if args.train_file != None:
    train = 1
    print("Train mode")

if args.test_file != None:
    predict = 1
    print("Predict mode")

if args.sentiment_embed_file != None:
    sentiment_vectors = load_embeddings(args.sentiment_embed_file, args.encoding)
    sentiment_embedding_weights = np.zeros((len(sentiment_vectors), args.embed_dim))

if args.sentiment_embed_file != None and (args.embed_file != None):
    print("Building sentiment embedding matrix")
    # Assumes that the sentiment embeddings and the pretrained embeddings (in args.embed_file)
    # are in the same space.
    pretrained_vectors = load_embeddings(args.embed_file, args.encoding)
    for key in sentiment_vectors:
        vec = sentiment_vectors[key]
        idx2sentvectors[sent_idx] = key
        sentiment_embedding_weights[sent_idx-1] = vec
        sent_idx +=1
        print(key, vec)
    sentiment_embedding_weights = np.transpose(sentiment_embedding_weights)

if args.cluster_file != None:
    word2clusteridx = read_clusters(args.cluster_file, args.encoding)

if train:
    print('Reading training data')
    params = dict()
    dataset_train = pd.read_csv(args.train_file, delimiter = '\t', header = None, encoding = args.encoding)

    training_set = dataset_train.iloc[:,:].values

    if args.label_encoding == "5-cls":
        print("Converting dataset to three labels (positive, negative, and neutral)")
        for i in range(len(training_set)):
            training_set[i,1] = fivecls2sent[int(training_set[i,1])]
    #training_set = training_set[:100]
    print("Number of samples", len(training_set))
    print("train positive:", len(training_set[training_set[:,-1] == 'positive']))
    print("train negative:", len(training_set[training_set[:,-1] == 'negative']))
    print("train neutral:", len(training_set[training_set[:,-1] == 'neutral']))

    # Number of unique tokens
    tokens = []
    for i in range(len(training_set)):
        print("train sentence:", i, training_set[i,0])
        words = training_set[i,0].strip().split()
        tokens.extend(words)
    token_dict = collections.Counter(tokens)
    max_tokens = {key: token_dict[key] for key in token_dict if token_dict[key] >= 5}
    tokens = set(tokens)

    print("Unique tokens:", len(tokens))
    print("Max tokens:", len(max_tokens))

    bilingual_dict = dict()
    mixed_tokens = dict()

    mixed_tokens = {key: max_tokens[key] for key in max_tokens}
    print("Number of mixed (total) tokens:", len(mixed_tokens))
    if args.lexicalize:
        bilingual_dict = read_bilingual_dict(args.bilingual_dict, args.encoding)
        # Add word translations to updatable embeddings
        for src_word in max_tokens:
            if src_word in bilingual_dict:
                trg_word = bilingual_dict[src_word]
                mixed_tokens[trg_word] = max_tokens[src_word]

    word2idx = OrderedDict()
    word2idx_updatable = OrderedDict()
    upd_idx = 1

    if args.updatable_embed:
        for key in mixed_tokens:
            word2idx_updatable[key] = upd_idx
            upd_idx +=1

    # Load embedding file
    if args.pretrain:
        print("Loading embedding matrix")
        pretrained_embed = load_embeddings(args.embed_file, args.encoding)
        embedding_matrix = np.zeros((len(pretrained_embed)+1, args.embed_dim))
        voc_idx = 1 # 0 is for words not in voc
        for key, vector in pretrained_embed.items():
            embedding_matrix[voc_idx] = vector
            word2idx[key] = voc_idx
            voc_idx += 1
        print("Voc idx", voc_idx) # There are duplicates probably for identical words (e.g punctuation)
        print("Embedding matrix shape:", embedding_matrix.shape)
        params['word2idx'] = word2idx

    if args.updatable_embed:
        print("Updatable dictionary length:", len(word2idx_updatable))
        params['word2idx_updatable'] = word2idx_updatable

    # Encode sequences
    X_train_seq = []
    fixed_seq_train = []
    updatable_seq_train = []
    sentiment_seq_train = []
    cluster_seq_train = []
    maxlen_fixed=0
    maxlen_updatable=0
    maxlen = 0
    transp_all =0
    wordsp_all = 0
    transu_all = 0
    wordsu_all = 0
    print('Converting data into sequences')

    for i in range(len(training_set)):
        print(i, training_set[i,0])
        if (i % 1000 == 0):
            print(i)
        if args.pretrain:
            seq, transp, wordsp = text_to_sequence(training_set[i,0], word2idx, args.lexicalize, bilingual_dict)
            fixed_seq_train.append(seq)
            transp_all += transp
            wordsp_all += wordsp
        if args.updatable_embed:
            seq, transu, wordsu = text_to_sequence(training_set[i,0], word2idx_updatable, args.lexicalize, bilingual_dict)
            updatable_seq_train.append(seq)
            transu_all += transu
            wordsu_all += wordsu
        if args.sentiment_embed_file != None:
            label_seq = sentiment_similarity(training_set[i,0], pretrained_embed, sentiment_vectors)
            sentiment_seq_train.append(label_seq)
        if args.cluster_file != None:
            seq, _, _  = text_to_sequence(training_set[i,0], word2clusteridx, args.lexicalize, bilingual_dict)
            cluster_seq_train.append(seq)


    if args.pretrain:
        fixed_seq_train = pad_sequences(np.array(fixed_seq_train))
        print("Fixed sequence shape:", fixed_seq_train.shape)
        maxlen_fixed = fixed_seq_train.shape[1]
        maxlen = maxlen_fixed
        print("% Translated using pretrained embeddings:", float(transp_all)/wordsp_all)
    if args.updatable_embed:
        updatable_seq_train = pad_sequences(np.array(updatable_seq_train))
        print("Updatable sequence shape:", updatable_seq_train.shape)
        maxlen_updatable = updatable_seq_train.shape[1]
        maxlen = maxlen_updatable
        print("% Translated using updatable embeddings:", float(transu_all)/wordsu_all)
    if args.sentiment_embed_file != None:
        sentiment_seq_train = pad_sequences(np.array(sentiment_seq_train))
        print("Sentiment sequence shape:", sentiment_seq_train.shape)
        print(sentiment_seq_train)
    if args.cluster_file != None:
        cluster_seq_train = pad_sequences(np.array(cluster_seq_train))
        print("Cluster sequence shape:", cluster_seq_train.shape)
        print(cluster_seq_train)

    params['maxlen_fixed'] = maxlen_fixed
    params['maxlen_updatable'] = maxlen_updatable
    params['maxlen'] = maxlen
    params['fixed'] = args.pretrain
    params['updatable'] = args.updatable_embed
    if args.pretrain:
        params['pretrained_embed'] = pretrained_embed
    params['sentiment'] = (args.sentiment_embed_file != None)
    params['clusters'] = (args.cluster_file != None)

    if args.sentiment_embed_file:
        params['sentiment_vectors'] = sentiment_vectors
    if args.cluster_file:
        params['word2clusteridx'] = word2clusteridx

    y_train = pd.get_dummies(training_set[:,-1])
    print("y_train shape", y_train.shape)
    print('maxlen_fixed:', maxlen_fixed)
    print('maxlen_updatable:', maxlen_updatable)

if predict:
    utf8 = 1
    dataset_test = pd.read_csv(args.test_file, delimiter = '\t', header = None, encoding = args.encoding)
    test_set = dataset_test.iloc[:,:].values


# Build LSTM network
if train:
    fixed_input = Input(shape=(maxlen_fixed,))
    updatable_input = Input(shape=(maxlen_updatable,))
    updatable_layer = Embedding(len(mixed_tokens)+1, args.embed_dim, input_length= maxlen_updatable)
    updatable_embeddings = updatable_layer(updatable_input)

    if args.pretrain:
        if args.update_pretrained == 1:
            embedding_layer = Embedding(embedding_matrix.shape[0], args.embed_dim, trainable = True, weights = [embedding_matrix], input_length = maxlen_fixed)
        else:
            embedding_layer = Embedding(embedding_matrix.shape[0], args.embed_dim, trainable = False, weights = [embedding_matrix], input_length = maxlen_fixed)
        fixed_embeddings = embedding_layer(fixed_input)
    if args.pretrain and args.updatable_embed:
        embed_in = concatenate([fixed_embeddings, updatable_embeddings])
        model_data = [fixed_seq_train, updatable_seq_train]
        model_inputs = [fixed_input, updatable_input]
    elif args.pretrain:
        print("Using only fixed embeddings")
        embed_in = fixed_embeddings
        model_data = fixed_seq_train
        model_inputs = fixed_input
    elif args.updatable_embed:
        print("Using only updatable embeddings")
        embed_in = updatable_embeddings
        model_data = updatable_seq_train
        model_inputs = updatable_input
    if args.cluster_file != None:
        '''only for when we have pretrained embeddings'''
        cluster_layer = Embedding(len(word2clusteridx)+1, args.cluster_dim, input_length = maxlen_fixed)
        cluster_input = Input(shape=(maxlen_fixed,))
        cluster_embeddings = cluster_layer(cluster_input)
        model_data = [fixed_seq_train, cluster_seq_train]
        model_inputs = [fixed_input, cluster_input]
        embed_in = concatenate([fixed_embeddings, cluster_embeddings])
    if args.sentiment_embed_file != None:
        sentiment_input = Input(shape=(maxlen_fixed,len(sentiment_vectors)))
        sentiment_seq_train = np.reshape(sentiment_seq_train, (sentiment_seq_train.shape[0], maxlen_fixed, len(sentiment_vectors)))
        print("sentiment seq train shape now:", sentiment_seq_train.shape)
        if args.cluster_file != None:
            model_inputs = [fixed_input, cluster_input, sentiment_input]
            model_data = [fixed_seq_train, cluster_seq_train, sentiment_seq_train]
        else:
            model_inputs = [fixed_input, sentiment_input]
            model_data = [fixed_seq_train, sentiment_seq_train]
        embed_in = concatenate([embed_in, sentiment_input])

    # default activation is tanh
    if args.bidirectional:
        lstm_out = Bidirectional(LSTM(units = args.hidden_units)) (embed_in)
    else:
        lstm_out = LSTM(units = args.hidden_units) (embed_in)

    # Avg pooling
    avg_pool_out = GlobalAveragePooling1D() (embed_in)

    # Final output
    dense_in = concatenate([lstm_out, avg_pool_out])
    dense_out = Dense(units = args.output_units, activation = 'softmax') (dense_in)
    model = Model(inputs = model_inputs, outputs = dense_out)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    print(model.summary())
    model.fit(model_data, y_train, batch_size = args.batch_size, epochs = args.epochs)
    print("Done training. Saving model.")
    model.save(os.path.join(args.out_dir, args.model_file))
    with open(os.path.join(args.out_dir, args.param_file), 'wb') as pf:
        pickle.dump(params, pf, protocol=pickle.HIGHEST_PROTOCOL)

# if test
else:
    model = load_model(os.path.join(args.out_dir, args.model_file))
    try:
        params = pickle.load(open(os.path.join(args.out_dir, args.param_file), 'rb'))
    except:
        params = dict()
    if 'word2idx' in params:
        word2idx = params['word2idx']
        maxlen_fixed = params['maxlen_fixed']

    if 'word2idx_updatable' in params:
        word2idx_updatable = params['word2idx_updatable']
        maxlen_updatable = params['maxlen_updatable']


# Predict and Evaluate
if predict:
    print("Preparing test sequence")
    X_test = []
    X_test_seq = []
    fixed_seq_test = []
    updatable_seq_test = []
    sentiment_seq_test = []
    cluster_seq_test = []

    updatable = params['updatable']
    fixed = params['fixed']
    clusters = params['clusters']
    sentiment = params['sentiment']
    if sentiment:
        sentiment_vectors = params['sentiment_vectors']
    if 'pretrained_embed' in params:
        pretrained_embed = params['pretrained_embed']
        if 'word2idx' not in params:
            word2idx = dict()
            voc_idx = 1
            for key, vector in pretrained_embed.items():
                word2idx[key] = voc_idx
                voc_idx += 1
    if 'word2idx_updatable' in params:
        word2idx_updatable = params['word2idx_updatable']
        maxlen_updatable = params['maxlen_updatable']
    if 'word2clusteridx' in params:
        word2clusteridx = params['word2clusteridx']
    if 'maxlen' in params:
        maxlen = params['maxlen']
    elif updatable:
        maxlen = maxlen_updatable
    elif fixed:
        maxlen = maxlen_fixed
   

    for i in range(len(test_set)):
        X_test.append(test_set[i,0].strip())
        if fixed:
            seq, _, _ = text_to_sequence(test_set[i,0], word2idx)
            fixed_seq_test.append(seq)
        if updatable:
            seq, _, _ = text_to_sequence(test_set[i,0], word2idx_updatable)
            updatable_seq_test.append(seq)
        if sentiment:
            label_seq = sentiment_similarity(test_set[i,0], pretrained_embed, sentiment_vectors)
            sentiment_seq_test.append(label_seq)
        if clusters:
            seq, _, _ = text_to_sequence(test_set[i,0], word2clusteridx)
            cluster_seq_test.append(seq)

    if fixed:
        fixed_seq_test = pad_sequences(np.array(fixed_seq_test), maxlen = maxlen_fixed)
        print("Fixed sequence test shape:", fixed_seq_test.shape)

    if updatable:
        updatable_seq_test = pad_sequences(np.array(updatable_seq_test), maxlen = maxlen_updatable)
        print("Updatable sequence test shape:", updatable_seq_test.shape)

    if sentiment:
        sentiment_seq_test = pad_sequences(np.array(sentiment_seq_test), maxlen = maxlen_fixed)
        print("Sentiment sequence test shape:", sentiment_seq_test.shape)

    if clusters:
        cluster_seq_test = pad_sequences(np.array(cluster_seq_test), maxlen = maxlen_fixed)
        print("Cluster sequence test shape:", cluster_seq_test.shape)


    if fixed and updatable:
        X_test_seq = [fixed_seq_test, updatable_seq_test]
    elif fixed:
        X_test_seq = fixed_seq_test
    elif updatable:
        X_test_seq = updatable_seq_test
    if clusters:
        X_test_seq = [fixed_seq_test, cluster_seq_test]
    if sentiment:
        if clusters:
            X_test_seq = [fixed_seq_test, cluster_seq_test, sentiment_seq_test]
        else:
            X_test_seq = [fixed_seq_test, sentiment_seq_test]


    y_test = pd.get_dummies(test_set[:,-1])
    if (y_test.shape[1] > args.output_units):
        print('error in test set shape! cutting extra columns')
        print("y_test shape", y_test.shape)
        print(y_test)
        y_test = y_test.iloc[:,:args.output_units].values
        print(y_test)

    print("Predicting")
    y_pred = model.predict(X_test_seq)
    print("y_pred shape", y_pred.shape)


    y_pred_sentiment = []
    y_pred_idx = []
    if args.has_labels:
        print('Evaluating')
        score, acc = model.evaluate(X_test_seq, y_test)
        print ("Score:", score, "Accuracy:", acc)
        y_test_idx = get_label_idx(y_test) # not using
    else:
        print('No labels in file')

    for i in range(len(y_pred)):
        pred = np.argmax(y_pred[i])
        y_pred_idx.append(pred)
        if args.has_labels:
            y_pred_sentiment.append(idx2sentiment[pred])
        else:
            y_pred_sentiment.append([idx2sentiment[pred], np.max(y_pred[i])])

    print('Done predicting')


    print('Writing to file')
    out_file = os.path.join(args.out_dir, args.out_file)
    print('Out file:', out_file)

    if args.has_labels:
        write_to_file(X_test, y_pred_sentiment, out_file, args.encoding)
    else:
        write_to_file_probs(X_test, y_pred_sentiment, out_file, args.encoding)
