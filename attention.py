 # Attention utils for cross-lingual targeted sentiment system
# See this paper by Liu and Zhang: Attention Modeling for Targeted Sentiment
# for the monolingual version (or a similar one to it)

from keras.layers import Dense, Input, Flatten, Activation, Average, Permute, RepeatVector, Lambda, Multiply, Concatenate, Subtract, Masking
from keras.layers import merge, concatenate
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import keras.backend.tensorflow_backend as K
import numpy as np

def attention_single_input(tensors):
    
    '''
    layer for target specific attention
    computes attention weights of inputs w.r.t to target
    inputs.shape = (batch_size, time_steps, input_dim)
    (None, 32) -> RepeatVector(3) -> (None, 3, 32)

    '''
    # Must import inside lambda function otherwise model won't load
    from keras.layers import Dense, Input, Flatten, Activation, Average, Permute, RepeatVector, Lambda, Multiply, Subtract, concatenate, merge, Masking
    import keras.backend.tensorflow_backend as K
    from attention import get_target_hidden_states, get_inputs

    inputs = tensors[0]
    trg_seq = tensors[1]
    units = int(inputs.shape[2])
    time_steps = K.int_shape(inputs)[1]

    # Get target hidden states
    trg_hidden_states = get_target_hidden_states(inputs, trg_seq) 
    trg_repeated = RepeatVector(time_steps)(trg_hidden_states)

    # Get inputs
    sentence_input = get_inputs(inputs, trg_seq)
    sentence_context = concatenate([sentence_input, trg_repeated], axis = 2)
    attn = Dense(1, activation='tanh') (sentence_context)
    attn = Flatten()(attn)
    attn = Activation('softmax')(attn)

    attn = RepeatVector(units)(attn)
    attn = Permute([2,1], name='attention_vec')(attn)
    s = merge([sentence_input,attn], name='attention_mul', mode='mul')
    return [s, attn]


def get_target_hidden_states(inputs, trg_seq, keyword_op='sum'):
    ''' 
    returns target hidden state from target sequence
    trg_seq is a vector of zeros and ones with ones at the target positions
    the hidden states are summed for multi-word targets (e.g 'ben affleck')
    if there are multiple target keywords, the default operation is currently 
    to sum the hidden states for the keywords. Currently we don't support multiple keywords.
    Currently, for multiple targets, it will return the sum of hidden states of
    target words. This can also be modified to return the average instead.
    axis 0: batch size, axis 1: time steps, axis 2: units
    '''
    units = int(inputs.shape[2])
    trg_seq = RepeatVector(units)(trg_seq)
    trg_seq = Permute([2,1])(trg_seq)
    trg_hidden_states = Multiply()([inputs, trg_seq]) # axis
    trg_hidden_states = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s:(s[0],s[2])) (trg_hidden_states)
    return trg_hidden_states


def get_inputs(inputs, trg_seq, context='full'):
    '''
    returns the sentence inputs to attention given the target sequence
    trg_seq is a vector of zeros and ones with ones at the target positions
    Currently, it will exclude the target words from the input
    and include the full sentence input. 
    The target words will be masked (replaced with zeros)
    This can also be modified to return left and right inputs.
    '''
    units = int(inputs.shape[2])
    ones= K.ones_like(trg_seq)
    trg_seq = Subtract()([ones, trg_seq])
    trg_seq = RepeatVector(units)(trg_seq)
    trg_seq = Permute([2,1])(trg_seq)
    new_inputs = Multiply()([inputs, trg_seq])
    return new_inputs





    