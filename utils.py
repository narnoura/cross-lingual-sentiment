# Utils for cross-lingual sentiment analysis

import codecs
import numpy as np
from scipy import spatial
from collections import OrderedDict
    
def cosine_similarity(v1, v2):
    if (v1==0 or v2==0):
        return 0.0
    else:
        return (1 - spatial.distance.cosine(v1, v2))

def load_embeddings(embed_file, encoding='utf-8'):
    ef = codecs.open(embed_file,'r',encoding)
    ef.readline()
    print('read first line')
    pretrained_embed = OrderedDict()
    ok=0
    err=0
    print('now reading embedding matrix')
    for line in ef:
        print(line)
        key = line.split(' ')[0]
        vec = [float(f) for f in line.strip().split(' ')[1:]]
        if len(vec)==0 or (len(vec)!=300 and len(vec)!=100):
            print('error reading vector')
            print(line.encode('utf-8'))
            err+=1
            continue
        else:
            ok+=1
        pretrained_embed[key] = vec
    ef.close()
    print("ok:", ok)
    print("err:",err)
    return pretrained_embed

def read_clusters(cluster_file, encoding='utf-8'):
    # read clusters from Brown cluster output file and returns word2cluster ids
    cf = codecs.open(cluster_file,'r',encoding)
    word2clusters = OrderedDict()
    prev_id = ""
    cluster_idx = 0
    for line in cf:
        try:
            cluster_id, word, freq = line.strip().split('\t')
        except:
            print("error parsing this line, skipping")
            print(line)
            continue
        if cluster_id != prev_id:
            cluster_idx +=1
            prev_id = cluster_id
        word2clusters[word] = cluster_idx
    return word2clusters

def read_bilingual_dict(dict_file, encoding):
    print('dict_file:', dict_file)
    bilingual_dict = dict()
    df = codecs.open(dict_file,'r', encoding)
    for line in df:
        try:
            src, trg = line.strip().split('\t', 2)
        except ValueError:
            continue
        bilingual_dict[src] = trg
    return bilingual_dict

def read_simple_sentiment_dict(sentiment_dict_file):
    ''' reads a simple tab separated sentiment lexicon'''
    sentiment_dict = dict()
    sf = codecs.open(sentiment_dict_file, 'r','utf-8')
    for line in sf:
        word, pol = line.strip().split('\t')
        sentiment_dict[word] = pol
    return sentiment_dict


def read_sentiment_dict(sentiment_dict_file):
    ''' reads the English MPQA subjectivity clues lexicon
    field names are: type (weaksubj/strongsubj), len, word1, pos1, stemmed1, and priorpolarity (positive/negative/neutral)
    This dictionary will be used to
    (1) filter words in updatable embedding matrix so that only subjective and sentiment bearing words are kept
    (2) if lexicalizing (with either pretrained or updatable embeddings), translate only subjective and sentiment bearing words
    '''
    sentiment_dict = dict()
    sf = codecs.open(sentiment_dict_file, 'r','utf-8')
    for line in sf:
        fields = line.strip().split(' ')
        word = fields[2].split('=', 2)[1]
        if not word in sentiment_dict:
            sentiment_dict[word] = {}
        for f in fields:
            print(f)
            field_name, value = f.split('=',2)
            sentiment_dict[word][field_name] = []
            sentiment_dict[word][field_name].append(value)

    return sentiment_dict

def read_sentiwordnet(sentiwordnet_file):
    '''
    reads sentiwordnet file and returns sentiwordnet score vectors for each word
    '''
    sw = codecs.open(sentiwordnet_file, 'r', 'utf-8')
    sentiwordnet_scores = dict()
    for line in sw:
        line = line.strip()
        word, pos_score, neg_score = line.split('\t')
        pos_score = float(pos_score)
        neg_score = float(neg_score)
        neut_score = 1 - pos_score - neg_score
        sentiwordnet_scores[word] = [pos_score, neg_score, neut_score]
        #print("word:", word, "sentiwordnet scores:", sentiwordnet_scores[word])
    return sentiwordnet_scores

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
