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

from util_layers import DenseLayer3DInput, Softmax3D, ApplyAttention, GatedEncoder3D
from dataset import SNLI

import pdb
theano.config.compute_test_value = 'warn'  # 'off' # Use 'warn' to activate this feature


LSTMHID = int(sys.argv[1])          # 150 Hidden unit numbers in LSTM
ATTHID = int(sys.argv[2])           # 350 Hidden unit numbers in attention MLP
OUTHID = int(sys.argv[3])           # 3000 Hidden unit numbers in output MLP
NROW = int(sys.argv[4])             # 10 Number of rows in matrix representation
LR = float(sys.argv[5])             # 0.01
L2REG = float(sys.argv[6])          # 0.0001 L2 regularization
DPOUT = float(sys.argv[7])          # 0.3 dropout rate
ATTPENALTY = float(sys.argv[8])     # 1.
WEDIM = int(sys.argv[9])            # 300 Dim of word embedding
BSIZE = int(sys.argv[10])           # 50 Minibatch size
GCLIP = float(sys.argv[11])         # 0.5 All gradients above this will be clipped
NEPOCH = int(sys.argv[12])          # 12 Number of epochs to train the net
STD = float(sys.argv[13])           # 0.1 Standard deviation of weights in initialization
UPDATEWE = bool(int(sys.argv[14]))  # 0 for False and 1 for True. Update word embedding in training
filename = __file__.split('.')[0] + \
           '_LSTMHID' + str(LSTMHID) + \
           '_ATTHID' + str(ATTHID) + \
           '_OUTHID' + str(OUTHID) + \
           '_NROWS' + str(NROW) + \
           '_LR' + str(LR) + \
           '_L2REG' + str(L2REG) + \
           '_DPOUT' + str(DPOUT) + \
           '_ATTPENALTY' + str(ATTPENALTY) + \
           '_WEDIM' + str(WEDIM) + \
           '_BSIZE' + str(BSIZE) + \
           '_GCLIP' + str(GCLIP) + \
           '_NEPOCH' + str(NEPOCH) + \
           '_STD' + str(STD) + \
           '_UPDATEWE' + str(UPDATEWE)

def main(num_epochs=NEPOCH):
    print("Loading data ...")
    snli = SNLI(batch_size=BSIZE)
    train_batches = list(snli.train_minibatch_generator())
    dev_batches = list(snli.dev_minibatch_generator())
    test_batches = list(snli.test_minibatch_generator())
    W_word_embedding = snli.weight  # W shape: (# vocab size, WE_DIM)
    del snli

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
        grad_clipping=GCLIP, backwards=True)
    
    # output dim: (BSIZE, None, 2*LSTMHID)
    l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward], axis=2)
    l_concat_dpout = lasagne.layers.DropoutLayer(l_concat, p=DPOUT, rescale=True)  # might not need this line

    # Attention mechanism to get sentence embedding
    # output dim: (BSIZE, None, ATTHID)
    l_ws1 = DenseLayer3DInput(l_concat_dpout, num_units=ATTHID)
    l_ws1_dpout = lasagne.layers.DropoutLayer(l_ws1, p=DPOUT, rescale=True)

    # output dim: (BSIZE, None, NROW)
    l_ws2 = DenseLayer3DInput(l_ws1_dpout, num_units=NROW, nonlinearity=None)
    l_annotations = Softmax3D(l_ws2, mask=l_mask)
    # output dim: (BSIZE, 2*LSTMHID, NROW)
    l_sentence_embedding = ApplyAttention([l_annotations, l_concat])

    # beam search? Bi lstm in the sentence embedding layer? etc.


    ########### get embeddings for hypothesis and premise ###########
    # hypothesis
    input_var_h = T.TensorType('int32', [False, False])('hypothesis_vector')
    input_var_h.tag.test_value = numpy.hstack((numpy.random.randint(1, 10000, (BSIZE, 18), 'int32'),
                                               numpy.zeros((BSIZE, 6)).astype('int32')))
    l_in_h = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_var_h)
    
    input_mask_h = T.TensorType('int32', [False, False])('hypo_mask')
    input_mask_h.tag.test_value = numpy.hstack((numpy.ones((BSIZE, 18), dtype='int32'),
                                                numpy.zeros((BSIZE, 6), dtype='int32')))
    input_mask_h.tag.test_value[1, 18:22] = 1
    l_mask_h = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_mask_h)
    
    # premise
    input_var_p = T.TensorType('int32', [False, False])('premise_vector')
    input_var_p.tag.test_value = numpy.hstack((numpy.random.randint(1, 10000, (BSIZE, 16), 'int32'),
                                               numpy.zeros((BSIZE, 3)).astype('int32')))
    l_in_p = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_var_p)
    
    input_mask_p = T.TensorType('int32', [False, False])('premise_mask')
    input_mask_p.tag.test_value = numpy.hstack((numpy.ones((BSIZE, 16), dtype='int32'),
                                                numpy.zeros((BSIZE, 3), dtype='int32')))
    input_mask_p.tag.test_value[1, 16:18] = 1
    l_mask_p = lasagne.layers.InputLayer(shape=(BSIZE, None), input_var=input_mask_p)
    
    
    hypothesis_embedding, hypothesis_annotation = lasagne.layers.get_output(
        [l_sentence_embedding, l_annotations],
        {l_in: l_in_h.input_var, l_mask: l_mask_h.input_var})
    premise_embedding, premise_annotation = lasagne.layers.get_output(
        [l_sentence_embedding, l_annotations],
        {l_in: l_in_p.input_var, l_mask: l_mask_p.input_var})

    hypothesis_embedding_clean, hypothesis_annotation_clean = lasagne.layers.get_output(
        [l_sentence_embedding, l_annotations],
        {l_in: l_in_h.input_var, l_mask: l_mask_h.input_var}, deterministic=True)
    premise_embedding_clean, premise_annotation_clean = lasagne.layers.get_output(
        [l_sentence_embedding, l_annotations],
        {l_in: l_in_p.input_var, l_mask: l_mask_p.input_var}, deterministic=True)

    ########### gated encoder and output MLP ##########
    l_hypo_embed = lasagne.layers.InputLayer(
        shape=(BSIZE, NROW, 2*LSTMHID), input_var=hypothesis_embedding)
    l_hypo_embed_dpout = lasagne.layers.DropoutLayer(l_hypo_embed, p=DPOUT, rescale=True)
    l_pre_embed = lasagne.layers.InputLayer(
        shape=(BSIZE, NROW, 2*LSTMHID), input_var=premise_embedding)
    l_pre_embed_dpout = lasagne.layers.DropoutLayer(l_pre_embed, p=DPOUT, rescale=True)
   
    # output dim: (BSIZE, NROW, 2*LSTMHID)
    l_factors = GatedEncoder3D([l_hypo_embed_dpout, l_pre_embed_dpout], num_hfactors=2*LSTMHID)
    l_factors_dpout = lasagne.layers.DropoutLayer(l_factors, p=DPOUT, rescale=True)
    
    # l_hids = DenseLayer3DWeight()

    l_outhid = lasagne.layers.DenseLayer(
        l_factors_dpout, num_units=OUTHID, nonlinearity=lasagne.nonlinearities.rectify)
    l_outhid_dpout = lasagne.layers.DropoutLayer(l_outhid, p=DPOUT, rescale=True)

    l_output = lasagne.layers.DenseLayer(
        l_outhid_dpout, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)


    ########### target, cost, validation, etc. ##########
    target_values = T.ivector('target_output')
    target_values.tag.test_value = numpy.asarray([1,] * BSIZE, dtype='int32')

    network_output = lasagne.layers.get_output(l_output)
    network_prediction = T.argmax(network_output, axis=1)
    accuracy = T.mean(T.eq(network_prediction, target_values))

    network_output_clean = lasagne.layers.get_output(
        l_output,
        {l_hypo_embed: hypothesis_embedding_clean,
         l_pre_embed: premise_embedding_clean},
        deterministic=True)
    network_prediction_clean = T.argmax(network_output_clean, axis=1)
    accuracy_clean = T.mean(T.eq(network_prediction_clean, target_values))
    
    # penalty term and cost
    attention_penalty = T.mean((T.batched_dot(
        hypothesis_annotation,
        hypothesis_annotation.dimshuffle(0, 2, 1)
    ) - T.eye(hypothesis_annotation.shape[1]).dimshuffle('x', 0, 1)
    )**2, axis=(0, 1, 2)) + T.mean((T.batched_dot(
        premise_annotation,
        premise_annotation.dimshuffle(0, 2, 1)
    ) - T.eye(premise_annotation.shape[1]).dimshuffle('x', 0, 1))**2, axis=(0, 1, 2))
   
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
    L2_attention = (l_ws1.W ** 2).sum() + (l_ws2.W ** 2).sum()
    L2_gae = (l_factors.Wxf ** 2).sum() + (l_factors.Wyf ** 2).sum()
    L2_outputhid = (l_outhid.W ** 2).sum()
    L2_softmax = (l_output.W ** 2).sum()
    L2 = L2_lstm + L2_attention + L2_gae + L2_outputhid + L2_softmax 

    cost = T.mean(T.nnet.categorical_crossentropy(network_output, target_values)) + \
           L2REG * L2
    cost_clean = T.mean(T.nnet.categorical_crossentropy(network_output_clean, target_values)) + \
                 L2REG * L2
    if ATTPENALTY != 0.:
        cost = cost + ATTPENALTY * attention_penalty
        cost_clean = cost_clean + ATTPENALTY * attention_penalty

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_output) + \
                 lasagne.layers.get_all_params(l_sentence_embedding)
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
        [l_in_h.input_var, l_mask_h.input_var,
         l_in_p.input_var, l_mask_p.input_var, target_values],
        [cost, accuracy], updates=updates)
    compute_cost = theano.function(
        [l_in_h.input_var, l_mask_h.input_var,
         l_in_p.input_var, l_mask_p.input_var, target_values],
        [cost_clean, accuracy_clean])
    predict = theano.function(
        [l_in_h.input_var, l_mask_h.input_var,
         l_in_p.input_var, l_mask_p.input_var],
        network_prediction_clean)

    def evaluate(mode, verbose=False):
        if mode == 'dev':
            data = dev_batches
        if mode == 'test':
            data = test_batches
        
        set_cost = 0.
        set_accuracy = 0.
        for batches_seen, (hypo, hm, premise, pm, truth) in enumerate(data, 1):
            _cost, _accuracy = compute_cost(hypo, hm, premise, pm, truth)
            set_cost = (1.0 - 1.0 / batches_seen) * set_cost + \
                       1.0 / batches_seen * _cost
            set_accuracy = (1.0 - 1.0 / batches_seen) * set_accuracy + \
                             1.0 / batches_seen * _accuracy
        
        if verbose == True:
            predicted = []
            truth = []
            for batches_seen, (hypo, hm, premise, pm, th) in enumerate(data, 1):
                predicted.append(predict(hypo, hm, premise, pm))
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
    print("BEFORE TRAINING: dev cost %f, accuracy %f" % (test_set_cost,  test_set_accuracy))
    print("Training ...")
    try:
        for epoch in range(num_epochs):
            train_set_cost = 0.
            train_set_accuracy = 0.
            start = time.time()
            
            for batches_seen, (hypo, hm, premise, pm, truth) in enumerate(
                    train_batches, 1):
                _cost, _accuracy = train(hypo, hm, premise, pm, truth)
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
                    print("***dev cost %f, accuracy %f" % (dev_set_cost,  dev_set_accuracy))

            # save parameters
            all_param_values = [p.get_value() for p in all_params]
            cPickle.dump(all_param_values,
                         open('params' + os.sep + 'params_' + filename + '.pkl', 'wb'))

            dev_set_cost,  dev_set_accuracy  = evaluate('dev')
            test_set_cost, test_set_accuracy = evaluate('test', verbose=True)

            print("epoch %d, cost: train %f dev %f test %f;\n"
                  "         accu: train %f dev %f test %f" % (
                epoch,
                train_set_cost,     dev_set_cost,     test_set_cost,
                train_set_accuracy, dev_set_accuracy, test_set_accuracy))
    except KeyboardInterrupt:
        pdb.set_trace()
        pass

if __name__ == '__main__':
    main()

