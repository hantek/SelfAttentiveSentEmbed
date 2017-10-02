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
from sklearn.metrics import confusion_matrix
import lasagne
from lasagne.layers.recurrent import Gate
from lasagne import init, nonlinearities

from util_layers import (DenseLayer3DInput, Softmax3D, ApplyAttention,
                         GatedEncoder3D, Maxpooling)
from dataset import YELP, AGE2

import pdb
theano.config.compute_test_value = 'off'  # 'off' # Use 'warn' to activate

"""
BEST test set result:
yelp    77.575% L2REG=0.0001,  DPOUT=0.3
age2    63.65%  L2REG=0.00001, DPOUT=0.2
"""
LSTMHID = int(sys.argv[1])          # 500 Hidden unit numbers in LSTM
OUTHID = int(sys.argv[2])           # 1000 Hidden unit numbers in output MLP
LR = float(sys.argv[3])             # 0.01 Smaller than 0.04.
L2REG = float(sys.argv[4])          # 0.0001 L2 regularization
DPOUT = float(sys.argv[5])          # 0.3 dropout rate
WE = str(sys.argv[6])               # either `word2vec` or `glove`
WEDIM = int(sys.argv[7])            # either  100       or  300   Dim
BSIZE = int(sys.argv[8])            # 16 Minibatch size
GCLIP = float(sys.argv[9])          # 0.5 All gradients above will be clipped
NEPOCH = int(sys.argv[10])          # 300 Number of epochs to train the net
STD = float(sys.argv[11])           # 0.1 Standard deviation of weights in init
                                    # very slightly better than 0.01
UPDATEWE = bool(int(sys.argv[12]))  # 1 0 for False and 1 for True. Update WE
DSET = str(sys.argv[13])            # dataset, either `yelp` or `age2`

filename = __file__.split('.')[0] + \
           '_LSTMHID' + str(LSTMHID) + \
           '_OUTHID' + str(OUTHID) + \
           '_LR' + str(LR) + \
           '_L2REG' + str(L2REG) + \
           '_DPOUT' + str(DPOUT) + \
           '_WE' + str(WE) + \
           '_WEDIM' + str(WEDIM) + \
           '_BSIZE' + str(BSIZE) + \
           '_GCLIP' + str(GCLIP) + \
           '_NEPOCH' + str(NEPOCH) + \
           '_STD' + str(STD) + \
           '_UPDATEWE' + str(UPDATEWE) + \
           '_DSET' + DSET

def main(num_epochs=NEPOCH):
    if DSET == 'yelp':
        print("Loading yelp dataset ...")
        loaded_dataset = YELP(
            batch_size=BSIZE,
            datapath="/home/hantek/datasets/NLC_data/yelp/word2vec_yelp.pkl")
    elif DSET == 'age2':
        print("Loading age2 dataset ...")
        loaded_dataset = AGE2(
            batch_size=BSIZE,
            datapath="/home/hantek/datasets/NLC_data/age2/word2vec_age2.pkl")
    else:
        raise ValueError("DSET was set incorrectly. Check your cmd args.")
    #                     yelp     age2
    # train data        500000    68450
    # dev/test data       2000     4000
    # vocab                      ~1.2e5
    # 

    train_batches = list(loaded_dataset.train_minibatch_generator())
    dev_batches = list(loaded_dataset.dev_minibatch_generator())
    test_batches = list(loaded_dataset.test_minibatch_generator())
    W_word_embedding = loaded_dataset.weight  # W shape: (# vocab size, WE_DIM)
    del loaded_dataset

    print("Building network ...")
    ########### sentence embedding encoder ###########
    # sentence vector, with each number standing for a word number
    input_var = T.TensorType('int32', [False, False])('sentence_vector')
    input_var.tag.test_value = numpy.hstack((numpy.random.randint(1, 10000, (BSIZE, 20), 'int32'),
                                             numpy.zeros((BSIZE, 5)).astype('int32')))
    input_var.tag.test_value[1, 20:22] = (413, 45)
    l_in = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_var)
    
    input_mask = T.TensorType('int32', [False, False])('sentence_mask')
    input_mask.tag.test_value = numpy.hstack((numpy.ones((BSIZE, 20), dtype='int32'),
                                             numpy.zeros((BSIZE, 5), dtype='int32')))
    input_mask.tag.test_value[1, 20:22] = 1
    l_mask = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_mask)

    # output shape (BSIZE, None, WEDIM)
    l_word_embed = lasagne.layers.EmbeddingLayer(
        l_in,
        input_size=W_word_embedding.shape[0],
        output_size=W_word_embedding.shape[1],
        W=W_word_embedding)

    # bidirectional LSTM
    l_forward = lasagne.layers.LSTMLayer(
        l_word_embed, mask_input=l_mask, num_units=LSTMHID,
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
        only_return_final=False,
        grad_clipping=GCLIP)

    l_backward = lasagne.layers.LSTMLayer(
        l_word_embed, mask_input=l_mask, num_units=LSTMHID,
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
        only_return_final=False,
        grad_clipping=GCLIP, backwards=True)
    
    # output dim: (BSIZE, None, 2*LSTMHID)
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)

    # output dim: (BSIZE, 2*LSTMHID)
    l_maxpool = Maxpooling(l_concat, axis=1)
    l_maxpool_dpout = lasagne.layers.DropoutLayer(l_maxpool, p=DPOUT, rescale=True)

    l_outhid = lasagne.layers.DenseLayer(
        l_maxpool_dpout, num_units=OUTHID,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_outhid_dpout = lasagne.layers.DropoutLayer(l_outhid, p=DPOUT, rescale=True)

    l_output = lasagne.layers.DenseLayer(
        l_outhid_dpout, num_units=5, nonlinearity=lasagne.nonlinearities.softmax)


    ########### target, cost, validation, etc. ##########
    target_values = T.ivector('target_output')
    target_values.tag.test_value = numpy.asarray([1,] * BSIZE, dtype='int32')

    network_output = lasagne.layers.get_output(l_output)
    network_prediction = T.argmax(network_output, axis=1)
    accuracy = T.mean(T.eq(network_prediction, target_values))

    network_output_clean = lasagne.layers.get_output(l_output, deterministic=True)
    network_prediction_clean = T.argmax(network_output_clean, axis=1)
    accuracy_clean = T.mean(T.eq(network_prediction_clean, target_values))
    
    L2_lstm = ((l_forward.W_in_to_ingate ** 2).sum() + \
               (l_forward.W_hid_to_ingate ** 2).sum() + \
               (l_forward.W_in_to_forgetgate ** 2).sum() + \
               (l_forward.W_hid_to_forgetgate ** 2).sum() + \
               (l_forward.W_in_to_cell ** 2).sum() + \
               (l_forward.W_hid_to_cell ** 2).sum() + \
               (l_forward.W_in_to_outgate ** 2).sum() + \
               (l_forward.W_hid_to_outgate ** 2).sum() + \
               (l_backward.W_in_to_ingate ** 2).sum() + \
               (l_backward.W_hid_to_ingate ** 2).sum() + \
               (l_backward.W_in_to_forgetgate ** 2).sum() + \
               (l_backward.W_hid_to_forgetgate ** 2).sum() + \
               (l_backward.W_in_to_cell ** 2).sum() + \
               (l_backward.W_hid_to_cell ** 2).sum() + \
               (l_backward.W_in_to_outgate ** 2).sum() + \
               (l_backward.W_hid_to_outgate ** 2).sum())
    L2_outputhid = (l_outhid.W ** 2).sum()
    L2_softmax = (l_output.W ** 2).sum()
    L2 = L2_lstm + L2_outputhid + L2_softmax 
    
    cost = T.mean(T.nnet.categorical_crossentropy(network_output,
                                                  target_values)) + \
           L2REG * L2
    cost_clean = T.mean(T.nnet.categorical_crossentropy(network_output_clean,
                                                        target_values)) + \
                 L2REG * L2

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_output)
    if not UPDATEWE:
        all_params.remove(l_word_embed.W)

    numparams = sum([numpy.prod(i) for i in [i.shape.eval() for i in all_params]])
    print("Number of params: {}\nName\t\t\tShape\t\t\tSize".format(numparams))
    print("-----------------------------------------------------------------")
    for item in all_params:
        print("{0:24}{1:24}{2}".format(item, item.shape.eval(), numpy.prod(item.shape.eval())))

    # if exist param file then load params
    look_for = 'params' + os.sep + 'params_' + filename + '.pkl'
    if os.path.isfile(look_for):
        print("Resuming from file: " + look_for)
        all_param_values = cPickle.load(open(look_for, 'rb'))
        for p, v in zip(all_params, all_param_values):
            p.set_value(v)
   
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LR)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function(
        [l_in.input_var, l_mask.input_var, target_values],
        [cost, accuracy], updates=updates)
    compute_cost = theano.function(
        [l_in.input_var, l_mask.input_var, target_values],
        [cost_clean, accuracy_clean])
    predict = theano.function(
        [l_in.input_var, l_mask.input_var],
        network_prediction_clean)

    def evaluate(mode, verbose=False):
        if mode == 'dev':
            data = dev_batches
        if mode == 'test':
            data = test_batches
        
        set_cost = 0.
        set_accuracy = 0.
        for batches_seen, (hypo, hm, truth) in enumerate(data, 1):
            _cost, _accuracy = compute_cost(hypo, hm, truth)
            set_cost = (1.0 - 1.0 / batches_seen) * set_cost + \
                       1.0 / batches_seen * _cost
            set_accuracy = (1.0 - 1.0 / batches_seen) * set_accuracy + \
                           1.0 / batches_seen * _accuracy
        if verbose == True:
            predicted = []
            truth = []
            for batches_seen, (sent, mask, th) in enumerate(data, 1):
                predicted.append(predict(sent, mask))
                truth.append(th)
            truth = numpy.concatenate(truth)
            predicted = numpy.concatenate(predicted)
            cm = confusion_matrix(truth, predicted)
            pr_a = cm.trace()*1.0 / truth.size
            pr_e = ((cm.sum(axis=0)*1.0/truth.size) * \
                    (cm.sum(axis=1)*1.0/truth.size)).sum()
            k = (pr_a - pr_e) / (1 - pr_e)
            print(mode + " set statistics:")
            print("kappa index of agreement: %f" % k)
            print("confusion matrix:")
            print(cm)

        return set_cost, set_accuracy
    

    print("Done. Evaluating scratch model ...")
    test_set_cost,  test_set_accuracy  = evaluate('test', verbose=True)
    print("BEFORE TRAINING: test cost %f, accuracy %f" % (
        test_set_cost, test_set_accuracy))
    print("Training ...")
    try:
        for epoch in range(num_epochs):
            train_set_cost = 0.
            train_set_accuracy = 0.
            start = time.time()
            
            for batches_seen, (hypo, hm, truth) in enumerate(train_batches, 1):
                _cost, _accuracy = train(hypo, hm, truth)
                train_set_cost = (1.0 - 1.0 / batches_seen) * train_set_cost + \
                                 1.0 / batches_seen * _cost
                train_set_accuracy = (1.0 - 1.0 / batches_seen) * train_set_accuracy + \
                                  1.0 / batches_seen * _accuracy
                if batches_seen % 100 == 0:
                    end = time.time()
                    print("Sample %d %.2fs, lr %.4f, train cost %f, accuracy %f"  % (
                        batches_seen * BSIZE,
                        end - start,
                        LR,
                        train_set_cost,
                        train_set_accuracy))
                    start = end

                if batches_seen % 2000 == 0:
                    dev_set_cost,  dev_set_accuracy = evaluate('dev')
                    test_set_cost, test_set_accuracy = evaluate('test')
                    print("RECORD: cost: train %f dev %f test %f\n"
                          "        accu: train %f dev %f test %f" % (
                        train_set_cost,     dev_set_cost,     test_set_cost,
                        train_set_accuracy, dev_set_accuracy, test_set_accuracy))

            # save parameters
            all_param_values = [p.get_value() for p in all_params]
            cPickle.dump(all_param_values,
                         open('params' + os.sep + 'params_' + filename + '.pkl', 'wb'))

            dev_set_cost,  dev_set_accuracy  = evaluate('dev')
            test_set_cost, test_set_accuracy = evaluate('test', verbose=True)
            print("RECORD:epoch %d, cost: train %f dev %f test %f\n"
                  "         accu: train %f dev %f test %f" % (
                epoch,
                train_set_cost,     dev_set_cost,     test_set_cost,
                train_set_accuracy, dev_set_accuracy, test_set_accuracy))
    except KeyboardInterrupt:
        pdb.set_trace()
        pass

if __name__ == '__main__':
    main()

