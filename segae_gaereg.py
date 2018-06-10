#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import os
import sys
import numpy
import cPickle
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.recurrent import Gate
from lasagne import init, nonlinearities

from util_layers import DenseLayer3DInput, Softmax3D, ApplyAttention, GatedEncoder3D
from dataset import SNLI

import pdb
theano.config.compute_test_value = 'warn'  # 'off' # Use 'warn' to activate this feature


LSTM_HIDDEN = int(sys.argv[1])          # 150 Hidden unit numbers in LSTM
ATTENTION_HIDDEN = int(sys.argv[2])     # 350 Hidden unit numbers in attention MLP
OUT_HIDDEN = int(sys.argv[3])           # 3000 Hidden unit numbers in output MLP
N_ROWS = int(sys.argv[4])               # 10 Number of rows in matrix representation
LEARNING_RATE = float(sys.argv[5])      # 0.01
ATTENTION_PENALTY = float(sys.argv[6])  # 1.
GAEREG = float(sys.argv[7])             # 0.5 Dropout in GAE
WE_DIM = int(sys.argv[8])               # 300 Dim of word embedding
BATCH_SIZE = int(sys.argv[9])           # 50 Minibatch size
GRAD_CLIP = int(sys.argv[10])           # 100 All gradients above this will be clipped
NUM_EPOCHS = int(sys.argv[11])          # 12 Number of epochs to train the net
STD = float(sys.argv[12])               # 0.1 Standard deviation of weights in initialization
filename = __file__.split('.')[0] + \
           '_LSTMHIDDEN' + str(LSTM_HIDDEN) + \
           '_ATTENTIONHIDDEN' + str(ATTENTION_HIDDEN) + \
           '_OUTHIDDEN' + str(OUT_HIDDEN) + \
           '_NROWS' + str(N_ROWS) + \
           '_LEARNINGRATE' + str(LEARNING_RATE) + \
           '_ATTENTIONPENALTY' + str(ATTENTION_PENALTY) + \
           '_GAEREG' + str(GAEREG) + \
           '_WEDIM' + str(WE_DIM) + \
           '_BATCHSIZE' + str(BATCH_SIZE) + \
           '_GRADCLIP' + str(GRAD_CLIP) + \
           '_NUMEPOCHS' + str(NUM_EPOCHS) + \
           '_STD' + str(STD)


def main(num_epochs=NUM_EPOCHS):
    print("Loading data ...")
    snli = SNLI(batch_size=BATCH_SIZE)
    train_batches = list(snli.train_minibatch_generator())
    dev_batches = list(snli.dev_minibatch_generator())
    test_batches = list(snli.test_minibatch_generator())
    W_word_embedding = snli.weight  # W shape: (# vocab size, WE_DIM)
    del snli

    print("Building network ...")
    ########### sentence embedding encoder ###########
    # sentence vector, with each number standing for a word number
    input_var = T.TensorType('int32', [False, False])('sentence_vector')
    input_var.tag.test_value = numpy.hstack((numpy.random.randint(1, 10000, (50, 20), 'int32'),
                                             numpy.zeros((50, 5)).astype('int32')))
    input_var.tag.test_value[1, 20:22] = (413, 45)
    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None), input_var=input_var)
    
    input_mask = T.TensorType('int32', [False, False])('sentence_mask')
    input_mask.tag.test_value = numpy.hstack((numpy.ones((50, 20), dtype='int32'),
                                             numpy.zeros((50, 5), dtype='int32')))
    input_mask.tag.test_value[1, 20:22] = 1
    l_mask = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None), input_var=input_mask)

    # output shape (BATCH_SIZE, None, WE_DIM)
    l_word_embed = lasagne.layers.EmbeddingLayer(
        l_in,
        input_size=W_word_embedding.shape[0],
        output_size=W_word_embedding.shape[1],
        W=W_word_embedding)  # how to set it to be non-trainable?


    # bidirectional LSTM
    l_forward = lasagne.layers.LSTMLayer(
        l_word_embed, mask_input=l_mask, num_units=LSTM_HIDDEN,
        ingate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD), 
                    W_cell=init.Normal(STD)),
        forgetgate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                    W_cell=init.Normal(STD)),
        cell=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                  W_cell=None, nonlinearity=nonlinearities.tanh),
        outgate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD), 
                    W_cell=init.Normal(STD)),
        nonlinearity=lasagne.nonlinearities.tanh,
        peepholes = False,
        grad_clipping=GRAD_CLIP)

    l_backward = lasagne.layers.LSTMLayer(
        l_word_embed, mask_input=l_mask, num_units=LSTM_HIDDEN,
        ingate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                    W_cell=init.Normal(STD)),
        forgetgate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                    W_cell=init.Normal(STD)),
        cell=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD),
                  W_cell=None, nonlinearity=nonlinearities.tanh),
        outgate=Gate(W_in=init.Normal(STD), W_hid=init.Normal(STD), 
                    W_cell=init.Normal(STD)),
        nonlinearity=lasagne.nonlinearities.tanh,
        peepholes = False,
        grad_clipping=GRAD_CLIP, backwards=True)
    
    # output dim: (BATCH_SIZE, None, 2*LSTM_HIDDEN)
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)

    # Attention mechanism to get sentence embedding
    # output dim: (BATCH_SIZE, None, ATTENTION_HIDDEN)
    l_ws1 = DenseLayer3DInput(l_concat, num_units=ATTENTION_HIDDEN)
    # output dim: (BATCH_SIZE, None, N_ROWS)
    l_ws2 = DenseLayer3DInput(l_ws1, num_units=N_ROWS, nonlinearity=None)
    l_annotations = Softmax3D(l_ws2, mask=l_mask)
    # output dim: (BATCH_SIZE, 2*LSTM_HIDDEN, N_ROWS)
    l_sentence_embedding = ApplyAttention([l_annotations, l_concat])

    # beam search? Bi lstm in the sentence embedding layer? etc.


    ########### get embeddings for hypothesis and premise ###########
    # hypothesis
    input_var_h = T.TensorType('int32', [False, False])('hypothesis_vector')
    input_var_h.tag.test_value = numpy.hstack((numpy.random.randint(1, 10000, (50, 18), 'int32'),
                                               numpy.zeros((50, 6)).astype('int32')))
    l_in_h = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None), input_var=input_var_h)
    
    input_mask_h = T.TensorType('int32', [False, False])('hypo_mask')
    input_mask_h.tag.test_value = numpy.hstack((numpy.ones((50, 18), dtype='int32'),
                                                numpy.zeros((50, 6), dtype='int32')))
    input_mask_h.tag.test_value[1, 18:22] = 1
    l_mask_h = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None), input_var=input_mask_h)
    
    # premise
    input_var_p = T.TensorType('int32', [False, False])('premise_vector')
    input_var_p.tag.test_value = numpy.hstack((numpy.random.randint(1, 10000, (50, 16), 'int32'),
                                               numpy.zeros((50, 3)).astype('int32')))
    l_in_p = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None), input_var=input_var_p)
    
    input_mask_p = T.TensorType('int32', [False, False])('premise_mask')
    input_mask_p.tag.test_value = numpy.hstack((numpy.ones((50, 16), dtype='int32'),
                                                numpy.zeros((50, 3), dtype='int32')))
    input_mask_p.tag.test_value[1, 16:18] = 1
    l_mask_p = lasagne.layers.InputLayer(shape=(BATCH_SIZE, None), input_var=input_mask_p)
    
    
    hypothesis_embedding, hypothesis_annotation = lasagne.layers.get_output(
        [l_sentence_embedding, l_annotations],
        {l_in: l_in_h.input_var, l_mask: l_mask_h.input_var})
    premise_embedding, premise_annotation = lasagne.layers.get_output(
        [l_sentence_embedding, l_annotations],
        {l_in: l_in_p.input_var, l_mask: l_mask_p.input_var})


    ########### gated encoder and output MLP ##########
    l_hypo_embed = lasagne.layers.InputLayer(
        shape=(BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN), input_var=hypothesis_embedding)
    l_pre_embed = lasagne.layers.InputLayer(
        shape=(BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN), input_var=premise_embedding)
   
    # output dim: (BATCH_SIZE, 2*LSTM_HIDDEN, N_ROWS)
    l_factors = GatedEncoder3D([l_hypo_embed, l_pre_embed], num_hfactors=2*LSTM_HIDDEN)

    # Dropout:
    l_factors_noise = lasagne.layers.DropoutLayer(l_factors, p=GAEREG, rescale=True)

    # l_hids = DenseLayer3DWeight()

    l_outhid = lasagne.layers.DenseLayer(
        l_factors_noise, num_units=OUT_HIDDEN, nonlinearity=lasagne.nonlinearities.rectify)

    # Dropout:
    l_outhid_noise = lasagne.layers.DropoutLayer(l_outhid, p=GAEREG, rescale=True)
    
    l_output = lasagne.layers.DenseLayer(
        l_outhid_noise, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)


    ########### target, cost, validation, etc. ##########
    target_values = T.ivector('target_output')
    target_values.tag.test_value = numpy.asarray([1,] * 50, dtype='int32')

    network_output = lasagne.layers.get_output(l_output)
    network_output_clean = lasagne.layers.get_output(l_output, deterministic=True)

    # penalty term and cost
    attention_penalty = T.mean((T.batched_dot(
        hypothesis_annotation,
        # pay attention to this line:
        # T.extra_ops.cpu_contiguous(hypothesis_annotation.dimshuffle(0, 2, 1))
        hypothesis_annotation.dimshuffle(0, 2, 1)
    ) - T.eye(hypothesis_annotation.shape[1]).dimshuffle('x', 0, 1)
    )**2, axis=(0, 1, 2)) + T.mean((T.batched_dot(
        premise_annotation,
        # T.extra_ops.cpu_contiguous(premise_annotation.dimshuffle(0, 2, 1))  # ditto.
        premise_annotation.dimshuffle(0, 2, 1)  # ditto.
    ) - T.eye(premise_annotation.shape[1]).dimshuffle('x', 0, 1))**2, axis=(0, 1, 2))
    
    cost = T.mean(T.nnet.categorical_crossentropy(network_output, target_values) + \
                  ATTENTION_PENALTY * attention_penalty)
    cost_clean = T.mean(T.nnet.categorical_crossentropy(network_output_clean, target_values) + \
                        ATTENTION_PENALTY * attention_penalty)

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_output) + \
                 lasagne.layers.get_all_params(l_sentence_embedding)
    numparams = sum([numpy.prod(i) for i in [i.shape.eval() for i in all_params]])
    print("Number of params: {}".format(numparams))
    
    # if exist param file then load params
    look_for = 'params' + os.sep + 'params_' + filename + '.pkl'
    if os.path.isfile(look_for):
        print("Resuming from file: " + look_for)
        all_param_values = cPickle.load(open(look_for, 'rb'))
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)

    # withoutwe_params = all_params + [l_word_embed.W]
    
    # Compute updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    network_prediction = T.argmax(network_output, axis=1)
    error_rate = T.mean(T.neq(network_prediction, target_values))
    network_prediction_clean = T.argmax(network_output_clean, axis=1)
    error_rate_clean = T.mean(T.neq(network_prediction_clean, target_values))
    
    train = theano.function(
        [l_in_h.input_var, l_mask_h.input_var,
         l_in_p.input_var, l_mask_p.input_var, target_values],
        [cost, error_rate], updates=updates)
    compute_cost = theano.function(
        [l_in_h.input_var, l_mask_h.input_var,
         l_in_p.input_var, l_mask_p.input_var, target_values],
        [cost_clean, error_rate_clean])

    def evaluate(mode):
        if mode == 'dev':
            data = dev_batches
        if mode == 'test':
            data = test_batches
        
        set_cost = 0.
        set_error_rate = 0.
        for batches_seen, (hypo, hm, premise, pm, truth) in enumerate(data, 1):
            _cost, _error = compute_cost(hypo, hm, premise, pm, truth)
            set_cost = (1.0 - 1.0 / batches_seen) * set_cost + \
                       1.0 / batches_seen * _cost
            set_error_rate = (1.0 - 1.0 / batches_seen) * set_error_rate + \
                             1.0 / batches_seen * _error
        
        return set_cost, set_error_rate
    
    dev_set_cost,  dev_set_error  = evaluate('dev')
    print("BEFORE TRAINING: dev cost %f, error %f" % (dev_set_cost,  dev_set_error))
    print("Training ...")
    try:
        for epoch in range(num_epochs):
            train_set_cost = 0.
            train_set_error = 0.
            start = time.time()
            
            for batches_seen, (hypo, hm, premise, pm, truth) in enumerate(
                    train_batches, 1):
                _cost, _error = train(hypo, hm, premise, pm, truth)
                train_set_cost = (1.0 - 1.0 / batches_seen) * train_set_cost + \
                                 1.0 / batches_seen * _cost
                train_set_error = (1.0 - 1.0 / batches_seen) * train_set_error + \
                                       1.0 / batches_seen * _error
                if batches_seen % 100 == 0:
                    end = time.time()
                    print("Sample %d %.2fs, lr %.4f, train cost %f, error %f"  % (
                        batches_seen * BATCH_SIZE,
                        LEARNING_RATE,
                        end - start,
                        train_set_cost,
                        train_set_error))
                    start = end

                if batches_seen % 2000 == 0:
                    dev_set_cost,  dev_set_error  = evaluate('dev')
                    test_set_cost, test_set_error = evaluate('test')
                    print("***dev  cost %f, error %f" % (dev_set_cost,  dev_set_error))
                    print("***test cost %f, error %f" % (test_set_cost,  test_set_error))

            # save parameters
            all_param_values = [p.get_value() for p in all_params]
            cPickle.dump(all_param_values,
                         open('params' + os.sep + 'params_' + filename + '.pkl', 'wb'))

            # load params
            # all_param_values = cPickle.load(open('params' + os.sep + 'params_' + filename, 'rb'))
            # for p, v in zip(all_params, all_param_values):
            #     p.set_value(v)

            dev_set_cost,  dev_set_error  = evaluate('dev')
            test_set_cost, test_set_error = evaluate('test')

            print("epoch %d, cost: train %f dev %f test %f;\n"
                  "         error train %f dev %f test %f" % (
                epoch,
                train_set_cost,     dev_set_cost,   test_set_cost,
                train_set_error,    dev_set_error,  test_set_error))
    except KeyboardInterrupt:
        pdb.set_trace()
        pass

if __name__ == '__main__':
    main()
