# Implements target-dependent and target context LSTMs (see )
#(for splitting into left and right sequences with respect to target)
# See this paper by Duyu Tang et al: Effective LSTMs for Target-Dependent Sentiment Classification
# for detailed description of TC-LSTM and TD-LSTM

from keras.layers import Dense, Input, Flatten, Activation, Average, Permute, RepeatVector, Lambda, Multiply, Concatenate, Subtract
from keras.layers import merge, concatenate
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
#from keras import backend as K
import keras.backend.tensorflow_backend as K
import numpy as np

def tc_lstm_input(tensors):
    '''
    Lambda function for TC-LSTM
    inputs.shape = (batch_size, time_steps, input_dim)
    '''
    
    from keras.layers import Permute, RepeatVector, Lambda, Multiply, Subtract, concatenate, merge, Masking
    import keras.backend.tensorflow_backend as K
    from tc_lstm import get_target_embeddings, get_inputs
    
    inputs = tensors[0]
    trg_seq = tensors[1]
    time_steps = K.int_shape(inputs)[1]
    trg_embeddings = get_target_embeddings(inputs, trg_seq)
    trg_repeated = RepeatVector(time_steps)(trg_embeddings)
    sentence_input = get_inputs(inputs, trg_seq)
    #sentence_input = inputs
    #sentence_input = Masking(mask_value=0.0) (sentence_input)
    sentence_input = concatenate([sentence_input, trg_repeated], axis = 2)
    return sentence_input

def td_lstm_input(tensors):
    '''
    Returns masked input for td_lstm for a left or right sequence
    Inputs with 0's in the input sequence will be zeroed out
    while inputs with 1's in the put sequence will be maintained
    '''
    from keras.layers import Permute, RepeatVector, Lambda, Multiply, Subtract, concatenate, merge, Masking
    import keras.backend.tensorflow_backend as K
    from tc_lstm import mask_inputs
    
    inputs = tensors[0]
    inp_seq = tensors[1]
    masked_inputs = mask_inputs(inputs, inp_seq)
    #masked_inputs = Masking(mask_value=0.0) (masked_inputs)
    return masked_inputs
    
def mask_inputs(inputs, inp_seq):
    ''' 
    masks inputs with 0's in input sequence 
    '''
    units = int(inputs.shape[2])
    inp_seq = RepeatVector(units)(inp_seq)
    inp_seq = Permute([2,1])(inp_seq)
    inp_embeddings = Multiply()([inputs, inp_seq]) 
    return inp_embeddings

def get_target_embeddings(inputs, trg_seq, keyword_op = 'mean'):
    '''
    Returns the mean of target embeddings to concatenate with the input
    for TC-LSTM.
    trg_seq is a vector of zeros and ones with ones at the target positions
    Embeddings will be returned at target positions and then averaged.
    '''
    
    units = int(inputs.shape[2])
    trg_seq = RepeatVector(units)(trg_seq)
    trg_seq = Permute([2,1])(trg_seq)
    trg_embeddings = Multiply()([inputs, trg_seq]) 
    trg_embeddings = Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s:(s[0],s[2])) (trg_embeddings)
    return trg_embeddings


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
    #ones = K.ones(shape=(None,[K.int_shape(inputs)[1]]))
    # flip zeroes to ones and ones to zeroes
    trg_seq = Subtract()([ones, trg_seq])
    trg_seq = RepeatVector(units)(trg_seq)
    trg_seq = Permute([2,1])(trg_seq)
    new_inputs = Multiply()([inputs, trg_seq])
    return new_inputs