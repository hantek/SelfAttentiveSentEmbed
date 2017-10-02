import numpy
import theano
import theano.tensor as T

from lasagne import nonlinearities, init
from lasagne.layers.base import Layer, MergeLayer

import pdb


class FlatConcat(MergeLayer):
    """
    ConCatLayer but Flattened to 2 dims before concatenation.
    Accepts more than 2 input. But all inputs should have the same dimention in
    the first dimention. This layer flattens all input to a 2-D matrix and
    concatenates them in the second dimention.

    """
    def get_output_shape_for(self, input_shapes):
        output_shapes = []
        for shape in input_shapes:
            output_shapes.append((shape[0], numpy.prod(shape[1:])))
        return (output_shapes[0][0], sum([i[-1] for i in output_shapes]))

    def get_output_for(self, inputs, **kwargs):
        inputs = [i.flatten(2) for i in inputs]
        return T.concatenate(inputs, axis=1)


class DenseLayerTensorDot(Layer):
    """
    multiply N 3D matrices along two dimensions of a 3D matrix, and produce a
    3D output. In batch training case, these setting corresponds to:
    
    Input shape:    (dim1, dim2, dim3, dim4)  # (BATCH_SIZE, num_inputslices, N_ROWS, num_inputfeatures)
    weight shape:   There are two type of weight dims:
                    'col': (num_slices, num_features, dim2, dim4)
                    'row': (num_slices, num_features, dim2, dim3)
    Output shape:   There are two types of output shapes:
                    'col': (dim1, num_slices, dim3, num_features)
                         # (BSIZE, num_slices, N_ROWS, num_features)
                    'row': (dim1, num_slices, num_features, num_inputfeatures)
                         # (BSIZE, num_slices, num_features, num_inputfeatures)

    direction: 'row': you are modifying along the row direction, thus the num_inputfeatures keeps intact.
            or 'col': you are modifying along the col direction (the number of features),
                      thus the N_ROWS will keep constant
    """
    def __init__(self, incoming, num_slices, num_features, direction='col',
                 W=init.GlorotUniform(gain='relu'), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(DenseLayerTensorDot, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.num_inputslices = self.input_shape[1]
        self.num_slices = num_slices
        self.num_inputfeatures = self.input_shape[3]
        self.num_features = num_features
        self.batch_size = self.input_shape[0]
        self.num_rows = self.input_shape[2]

        self.direction = direction
        if direction == 'col':
            self.W = self.add_param(
                W,
                (num_slices, num_features, self.num_inputslices, self.num_inputfeatures),
                name="W4D_TensorDot_col")
            self.axes = [[1, 3], [2, 3]]
        elif direction == 'row':
            self.W = self.add_param(
                W,
                (num_slices, num_features, self.num_inputslices, self.num_rows),
                name="W4D_TensorDot_row")
            self.axes = [[1, 2], [2, 3]]
        else:
            raise ValueError("`direction` has to be either `row` or `col`.")

    def get_output_shape_for(self, input_shape):
        num_inputfeatures = input_shape[3]
        batch_size = input_shape[0]
        num_rows = input_shape[2]

        # this may change according to the dims you choose to multiply
        if self.direction == 'col':
            return (batch_size, self.num_slices, num_rows, self.num_features)
        elif self.direction == 'row':
            return (batch_size, self.num_slices, self.num_features, num_inputfeatures)
        
    def get_output_for(self, input, **kwargs):
        x = input
        if self.direction == 'col':
            preactivation = T.tensordot(x, self.W, axes=self.axes).dimshuffle(0, 2, 1, 3)
        elif self.direction == 'row':
            preactivation = T.tensordot(x, self.W, axes=self.axes).dimshuffle(0, 2, 3, 1)
        return self.nonlinearity(preactivation)


class DenseLayerTensorBatcheddot(Layer):
    """
    """
    def __init__(self):
        pass
    def get_output_shape_for(self):
        pass
    def get_output_for(self):
        pass


class DenseLayer3DWeight(Layer):
    """
    Apply a 3D matrix to a 3D input, basically it is just batched dot.

    Input: (BATCH_SIZE, inputs_per_row, N_ROWS)

    Weight: 
    Depending on whether the weight is multiplied from left side of input,
    there are two shapes:
        right multiply case: (N_ROWS, inputs_per_row, units_per_row)
        left multiply case:  (inputs_per_row, N_ROWS, units_per_row)

    Output:
        right multiply case: (BATCH_SIZE, units_per_row, N_ROWS)
        left multiply case:  (BATCH_SIZE, inputs_per_row, units_per_row)
    
    Params:
        incoming,
        units_per_row,
        W
        b
        leftmul : True if the weight is left multiplied to the input.
        nonlinearity
        **kwargs
    """
    def __init__(self, incoming, units_per_row, W=init.GlorotUniform(),
                 b=init.Constant(0.), leftmul=False, nonlinearity=nonlinearities.tanh,
                 **kwargs):
        super(DenseLayer3DWeight, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.units_per_row = units_per_row
        self.inputs_per_row = self.input_shape[1]
        self.num_rows = self.input_shape[2]
        self.leftmul = leftmul
        
        if leftmul:
            self.W = self.add_param(
                W, (self.inputs_per_row, self.num_rows, self.units_per_row), name='W3D')
        else:
            self.W = self.add_param(
                W, (self.num_rows, self.inputs_per_row, self.units_per_row), name='W3D')
        
        if b is None:
            self.b = None
        else:
            if self.leftmul:
                b = theano.shared(
                    numpy.zeros((1, self.inputs_per_row, self.units_per_row),
                                dtype=theano.config.floatX),
                    broadcastable=(True, False, False), 
                    name="b3D")
                self.b = self.add_param(spec=b,
                                        shape=(1, self.inputs_per_row, self.units_per_row),
                                        regularizable=False)
            else:
                b = theano.shared(
                    numpy.zeros((1, self.units_per_row, self.num_rows),
                                dtype=theano.config.floatX),
                    broadcastable=(True, False, False), 
                    name="b3D")
                self.b = self.add_param(spec=b,
                                        shape=(1, self.units_per_row, self.num_rows),
                                        regularizable=False)

    def get_output_shape_for(self, input_shape):
        if self.leftmul:
            return (input_shape[0], input_shape[1], self.units_per_row)
        else:
            return (input_shape[0], self.units_per_row, input_shape[2])

    def get_output_for(self, input, **kwargs):
        if self.leftmul:
            preact = T.batched_dot(T.extra_ops.cpu_contiguous(input.dimshuffle(1, 0, 2)),
                                   self.W).dimshuffle(1, 0, 2)
        else:
            preact = T.batched_dot(T.extra_ops.cpu_contiguous(input.dimshuffle(2, 0, 1)),
                                   self.W).dimshuffle(1, 2, 0)
        if self.b is not None:
            preact = preact + self.b
        return self.nonlinearity(preact)


class DenseLayer3DInput(Layer):
    """
    Apply a 2D matrix to a 3D input, so its a batched dot with shared slices.
    
    Input: (BATCH_SIZE, inputdim1, inputdim2)

    Weight: 
    Depending on whether the weight is multiplied from left side of input,
    there are two shapes:
        right multiply case: (inputdim2, num_units)

    Output:
    
    Params:
        incoming,
        units_per_row,
        W
        b
        leftmul : True if the weight is left multiplied to the input.
        nonlinearity
        **kwargs
    """
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.tanh,
                 **kwargs):
        super(DenseLayer3DInput, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = self.input_shape[2]

        self.W = self.add_param(W, (num_inputs, num_units), name="W2D")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b2D",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_units)

    def get_output_for(self, input, **kwargs):
        
        # pdb.set_trace()

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 'x', 0)
        return self.nonlinearity(activation)


class Softmax3D(MergeLayer):
    """Softmax is conducted on the middle dimension of a 3D tensor."""
    def __init__(self, incoming, mask=None, **kwargs):
        """
        mask: a lasagne layer.
        """
        incomings = [incoming]
        self.have_mask = False
        if mask:
            incomings.append(mask)
            self.have_mask = True
        super(Softmax3D, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        preactivations = inputs[0]
        if self.have_mask:
            mask = inputs[1]
            preactivations = \
                preactivations * mask.dimshuffle(0, 1, 'x').astype(theano.config.floatX) - \
                numpy.asarray(1e36).astype(theano.config.floatX) * \
                (1 - mask).dimshuffle(0, 1, 'x').astype(theano.config.floatX)
            
        annotation = T.nnet.softmax(
            preactivations.dimshuffle(0, 2, 1).reshape((
                preactivations.shape[0] * preactivations.shape[2],
                preactivations.shape[1]))
        ).reshape((
            preactivations.shape[0],
            preactivations.shape[2],
            preactivations.shape[1]
        )).dimshuffle(0, 2, 1)
        return annotation


class ApplyAttention(MergeLayer):
    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2], input_shapes[1][2])

    def get_output_for(self, inputs, **kwargs):
        annotation, sentence = inputs[0], inputs[1]
        return T.batched_dot(sentence.dimshuffle(0, 2, 1), annotation).dimshuffle(0, 2, 1)


class AugmentFeature(MergeLayer):
    """
    Input:
    x: (BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN)
    y: (BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN)

    Output: (BATCH_SIZE, N_ROWS, 8*LSTM_HIDDEN)
    """
    def get_output_shape_for(self, input_shapes):
        assert input_shapes[0] == input_shapes[1], (
            "The two input to AugmentFeature layer should have the same shape.")
        batch_size = input_shapes[0][0]
        num_rows = input_shapes[0][1]
        num_dim = input_shapes[0][2]
        return (batch_size, num_rows, 4 * num_dim)
    
    def get_output_for(self, inputs, **kwargs):
        x, y = inputs[0], inputs[1]
        return T.concatenate([x, y, x - y, x * y], axis=2)


class GatedEncoder3D(MergeLayer):
    """
    An implementation of the encoder part of a 3D Gated Autoencoder. It has
    the encoder only. 
    
    It just returns the factor of H, not H. To get the real H, add
    another dense layer on top of the output.

    See __paper__ for more info.

    Input:
    x: (BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN)
    y: (BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN)

    Output:
    hfactors = (BATCH_SIZE, N_ROWS, num_hfactors)
    
    """
    def __init__(self, incomings, num_hfactors,
                 Wxf=init.GlorotUniform(),
                 Wyf=init.GlorotUniform(),
                 **kwargs):
        super(GatedEncoder3D, self).__init__(incomings, **kwargs)
        self.num_xfactors = self.input_shapes[0][2]
        self.num_yfactors = self.input_shapes[1][2]
        self.num_rows = self.input_shapes[0][1]
        self.num_hfactors = num_hfactors
        self.Wxf = self.add_param(
            Wxf, (self.num_rows, self.num_xfactors, self.num_hfactors), name='Wxf')
        self.Wyf = self.add_param(
            Wyf, (self.num_rows, self.num_yfactors, self.num_hfactors), name='Wyf')

    def get_output_shape_for(self, input_shapes):
        batch_size = input_shapes[0][0]
        return (batch_size, self.num_rows, self.num_hfactors)

    def get_output_for(self, inputs, **kwargs):
        x, y = inputs[0], inputs[1]
        # xfactor = T.batched_dot(x.dimshuffle(2, 0, 1), self.Wxf).dimshuffle(1, 2, 0)
        # yfactor = T.batched_dot(y.dimshuffle(2, 0, 1), self.Wyf).dimshuffle(1, 2, 0)
        xfactor = T.batched_dot(
            T.extra_ops.cpu_contiguous(x.dimshuffle(1, 0, 2)), self.Wxf).dimshuffle(1, 0, 2)
        yfactor = T.batched_dot(
            T.extra_ops.cpu_contiguous(y.dimshuffle(1, 0, 2)), self.Wyf).dimshuffle(1, 0, 2)
        return xfactor * yfactor


class StackedGatedEncoder3D(MergeLayer):
    """
    An implementation of the encoder part of a 3D Gated Autoencoder. It has
    the encoder only. 
    
    It just returns the factor of H, not H. To get the real H, add
    another dense layer on top of the output.

    See __paper__ for more info.

    Input:
    x: (BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN)
    y: (BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN)

    Output:
    hfactors = (BATCH_SIZE, N_ROWS, num_hfactors)
    
    """
    def __init__(self, incomings,
                 Wxf1=init.GlorotUniform(),
                 Wyf1=init.GlorotUniform(),
                 Wxf2=init.GlorotUniform(),
                 Wyf2=init.GlorotUniform(),
                 **kwargs):
        super(StackedGatedEncoder3D, self).__init__(incomings, **kwargs)
        self.num_xfactors = self.input_shapes[0][2]
        self.num_yfactors = self.input_shapes[1][2]
        assert self.num_xfactors == self.num_yfactors
        self.num_rows = self.input_shapes[0][1]
        self.Wxf1 = self.add_param(
            Wxf1, (self.num_rows, self.num_xfactors, self.num_xfactors), name='Wxf1')
        self.Wyf1 = self.add_param(
            Wyf1, (self.num_rows, self.num_yfactors, self.num_yfactors), name='Wyf1')
        self.Wxf2 = self.add_param(
            Wxf2, (self.num_rows, self.num_xfactors, self.num_xfactors), name='Wxf2')
        self.Wyf2 = self.add_param(
            Wyf2, (self.num_rows, self.num_yfactors, self.num_yfactors), name='Wyf2')

    def get_output_shape_for(self, input_shapes):
        batch_size = input_shapes[0][0]
        return (batch_size, self.num_rows, self.num_xfactors)

    def get_output_for(self, inputs, **kwargs):
        x, y = inputs[0], inputs[1]
        # xfactor = T.batched_dot(x.dimshuffle(2, 0, 1), self.Wxf).dimshuffle(1, 2, 0)
        # yfactor = T.batched_dot(y.dimshuffle(2, 0, 1), self.Wyf).dimshuffle(1, 2, 0)
        xfactor1 = T.tanh(T.batched_dot(
            T.extra_ops.cpu_contiguous(x.dimshuffle(1, 0, 2)), self.Wxf1).dimshuffle(1, 0, 2))
        yfactor1 = T.tanh(T.batched_dot(
            T.extra_ops.cpu_contiguous(y.dimshuffle(1, 0, 2)), self.Wyf1).dimshuffle(1, 0, 2))
        xfactor2 = T.batched_dot(
            T.extra_ops.cpu_contiguous(xfactor1.dimshuffle(1, 0, 2)), self.Wxf2).dimshuffle(1, 0, 2)
        yfactor2 = T.batched_dot(
            T.extra_ops.cpu_contiguous(yfactor1.dimshuffle(1, 0, 2)), self.Wyf2).dimshuffle(1, 0, 2)
        return xfactor2 * yfactor2


class GatedEncoder3DSharedW(MergeLayer):
    """
    An implementation of the encoder part of a 3D Gated Autoencoder.

    It has the encoder only. 
    
    It just returns the factor of H, not H. To get the real H, add
    another dense layer on top of the output.

    See __paper__ for more info.
    
    the two inputs, x and y, have to have the same shape.

    """
    def __init__(self, incomings, num_hfactors,
                 Wf=init.GlorotUniform(),
                 **kwargs):
        super(GatedEncoder3DSharedW, self).__init__(incomings, **kwargs)
        self.num_factors = self.input_shapes[0][1]
        self.num_rows = self.input_shapes[0][2]
        self.num_hfactors = num_hfactors
        self.Wf = self.add_param(
            Wf, (self.num_rows, self.num_factors, self.num_hfactors), name='Wf')

    def get_output_shape_for(self, input_shapes):
        batch_size = input_shapes[0][0]
        return (batch_size, self.num_hfactors, self.num_rows)

    def get_output_for(self, inputs, **kwargs):
        x, y = inputs[0], inputs[1]
        # xfactor = T.batched_dot(x.dimshuffle(2, 0, 1), self.Wxf).dimshuffle(1, 2, 0)
        # yfactor = T.batched_dot(y.dimshuffle(2, 0, 1), self.Wyf).dimshuffle(1, 2, 0)
        xfactor = T.batched_dot(T.extra_ops.cpu_contiguous(x.dimshuffle(2, 0, 1)), self.Wf).dimshuffle(1, 2, 0)
        yfactor = T.batched_dot(T.extra_ops.cpu_contiguous(y.dimshuffle(2, 0, 1)), self.Wf).dimshuffle(1, 2, 0)
        return xfactor * yfactor


class GatedEncoder4D(MergeLayer):
    """
    An implementation of the encoder part of a 4D Gated Autoencoder.

    It has the encoder only. 
    
    It just returns the factor of H, not H. To get the real H, add
    another dense layer on top of the output.

    the two inputs, x and y, have to have the same shape.
    
    Input shape:    (dim1, dim2, num_factors)               # (BATCH_SIZE, N_ROWS, 2*LSTM_HIDDEN)
    weight shape:   (num_slices, num_factors, num_hfactors) # (N_SLICES, 2*LSTM_HIDDEN, num_hfactors)
    Output shape:   (dim1, num_slices, dim2, num_hfactors)  # (BATCH_SIZE, N_SLICES, N_ROWS, num_hfactors)

    """
    def __init__(self, incomings, num_slices, num_hfactors,
                 Wf=init.GlorotUniform(),
                 **kwargs):
        super(GatedEncoder4D, self).__init__(incomings, **kwargs)
        self.num_slices = num_slices
        self.num_factors = self.input_shapes[0][2]
        self.num_rows = self.input_shapes[0][1]
        self.num_hfactors = num_hfactors
        self.Wf = self.add_param(
            Wf, (self.num_slices, self.num_factors, self.num_hfactors), name='Wf')

    def get_output_shape_for(self, input_shapes):
        batch_size = input_shapes[0][0]
        return (batch_size, self.num_slices, self.num_rows, self.num_hfactors)

    def get_output_for(self, inputs, **kwargs):
        x, y = inputs[0], inputs[1]
        xfactor = T.tensordot(x, self.Wf, axes=(2, 1)).dimshuffle(0, 2, 1, 3)
        yfactor = T.tensordot(y, self.Wf, axes=(2, 1)).dimshuffle(0, 2, 1, 3)
        return xfactor * yfactor


class APAttentionBatch(MergeLayer):
    """
    Attention Pooling mechanism. Compute a normalized weight over input sentences Q and A.

    input: Q & A:     (BSIZE, dim1(dim2), DIM)
           Q & A mask (BSIZE, dim1(dim2))
    U:                (NROW, DIM, DIM)
    output: G:        (BSIZE, NROW, dim1, dim2)
    """
    def __init__(self, incomings, masks=None, num_row=10, init_noise=0.001, **kwargs):
        self.have_mask = False
        if masks:
            incomings = incomings + masks
            self.have_mask = True
        super(APAttentionBatch, self).__init__(incomings, **kwargs)
        self.num_row = num_row
        self.init_noise = init_noise
        self.num_dim = self.input_shapes[0][2]
        U = (numpy.identity(self.num_dim) + init.Normal(std=self.init_noise).sample(
                 shape=(self.num_row, self.num_dim, self.num_dim))
            ).astype(theano.config.floatX)
        self.U = self.add_param(U, U.shape, name='U')

    def get_output_shape_for(self, input_shapes):
        batch_size = input_shapes[0][0]
        num_wordQ = input_shapes[0][1]
        num_wordA = input_shapes[1][1]
        return (batch_size, self.num_row, num_wordQ, num_wordA)

    def get_output_for(self, inputs, **kwargs):
        Q = inputs[0]
        A = inputs[1]
        QU = T.tensordot(Q, self.U, axes=[2, 1])  # (BSIZE, dim1, NROW, DIM)
        QUA = T.batched_tensordot(QU, A, axes=[3, 2]).dimshuffle(0, 2, 1, 3)
        G = T.tanh(QUA)  # (BSIZE, NROW, dim1, dim2)

        if self.have_mask:
            Qmask = inputs[2]
            Amask = inputs[3]
            Gmask = T.batched_dot(Qmask.dimshuffle(0, 1, 'x'),
                                  Amask.dimshuffle(0, 'x', 1)).dimshuffle(0, 'x', 1, 2)
            G = G * Gmask - (1 - Gmask)  # pad -1 to trailing spaces.
        
        return G


class ComputeEmbeddingPool(MergeLayer):
    """
    Input :
        x: (BSIZE, NROW, DIM)
        y: (BSIZE, NROW, DIM)
    Output :
        (BSIZE, NROW, NROW)
    """
    def __init__(self, incomings, **kwargs):
        super(ComputeEmbeddingPool, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        xshape = input_shapes[0]
        yshape = input_shapes[1]
        return (xshape[0], xshape[1], yshape[1])

    def get_output_for(self, inputs, **kwargs):
        x = inputs[0]
        y = inputs[1]
        return T.batched_dot(x, y.dimshuffle(0, 2, 1))


class AttendOnEmbedding(MergeLayer):
    """
    incomings=[x, embeddingpool], masks=[xmask, ymask], direction='col'
    or
              [y, embeddingpool], masks=[xmask, ymask], direction='row'
    
    Output :
              alpha; or beta
    """
    def __init__(self, incomings, masks=None, direction='col', **kwargs):
        self.have_mask = False
        if masks:
            incomings = incomings + masks
            self.have_mask = True
        super(AttendOnEmbedding, self).__init__(incomings, **kwargs)
        self.direction = direction

    def get_output_shape_for(self, input_shapes):
        sent_shape = input_shapes[0]
        emat_shape = input_shapes[1]
        if self.direction == 'col':
            # x:    (BSIZE, R_x, DIM)
            # emat: (BSIZE. R_x, R_y)
            # out:  (BSIZE, R_y, DIM)
            return (sent_shape[0], emat_shape[2], sent_shape[2])
        elif self.direction == 'row':
            # y:    (BSIZE, R_y, DIM)
            # emat: (BSIZE. R_x, R_y)
            # out:  (BSIZE, R_x, DIM)
            return (sent_shape[0], emat_shape[1], sent_shape[2])

    def get_output_for(self, inputs, **kwargs):
        sentence = inputs[0]
        emat = inputs[1]
        if self.have_mask:
            xmask = inputs[2]
            ymask = inputs[3]
            xymask = T.batched_dot(xmask.dimshuffle(0, 1, 'x'),
                                   ymask.dimshuffle(0, 'x', 1))
            emat = emat * xymask.astype(theano.config.floatX) - \
                   numpy.asarray(1e36).astype(theano.config.floatX) * \
                   (1 - xymask).astype(theano.config.floatX)

        if self.direction == 'col':  # softmax on x's dim, and multiply by x
            annotation = T.nnet.softmax(
                emat.dimshuffle(0, 2, 1).reshape((
                    emat.shape[0] * emat.shape[2], emat.shape[1]))
            ).reshape((
                emat.shape[0], emat.shape[2], emat.shape[1]
            ))  # (BSIZE, R_y, R_x)
            if self.have_mask:
                annotation = annotation * ymask.dimshuffle(
                    0, 1, 'x').astype(theano.config.floatX)
        elif self.direction == 'row':  # softmax on y's dim, and multiply by y
            annotation = T.nnet.softmax(
                emat.reshape((
                    emat.shape[0] * emat.shape[1], emat.shape[2]))
            ).reshape((
                emat.shape[0], emat.shape[1], emat.shape[2]
            ))  # (BSIZE, R_x, R_y)
            if self.have_mask:
                annotation = annotation * xmask.dimshuffle(
                    0, 1, 'x').astype(theano.config.floatX)
        return T.batched_dot(annotation, sentence)


class MeanOverDim(MergeLayer):
    """
    dim can be a number or a tuple of numbers to indicate which dim to compute mean.
    """
    def __init__(self, incoming, mask=None, dim=1, **kwargs):
        incomings = [incoming]
        self.have_mask = False
        if mask:
            incomings.append(mask)
            self.have_mask = True
        super(MeanOverDim, self).__init__(incomings, **kwargs)
        self.dim = dim

    def get_output_shape_for(self, input_shapes):
        return tuple(x for i, x in enumerate(input_shapes[0]) if i != self.dim)

    def get_output_for(self, inputs, **kwargs):
        if self.have_mask:
            return T.sum(inputs[0], axis=self.dim) / \
                   inputs[1].sum(axis=1).dimshuffle(0, 'x')
        else:
            return T.mean(inputs[0], axis=self.dim)


class MaxpoolingG(Layer):
    """
    Input : G matrix,
    Input shape: (BSIZE, NROW, dim1, dim2)

    Output shape:
        'row': (BSIZE, dim2, NROW)
        'col': (BSIZE, dim1, NROW)
    """
    def __init__(self, incoming, direction='col', **kwargs):
        super(MaxpoolingG, self).__init__(incoming, **kwargs)
        self.direction = direction

    def get_output_shape_for(self, input_shape):
        if self.direction == 'row':
            return (input_shape[0], input_shape[3], input_shape[1])
        elif self.direction == 'col':
            return (input_shape[0], input_shape[2], input_shape[1])

    def get_output_for(self, input, **kwargs):
        G = input
        if self.direction == 'row':
            return T.max(G, axis=2).dimshuffle(0, 2, 1)
        elif self.direction == 'col':
            return T.max(G, axis=3).dimshuffle(0, 2, 1)


class Maxpooling(Layer):
    """
    Input : N-D matrix,
    Input shape: (BSIZE, NROW, dim1, dim2)

    Output shape:
    """
    def __init__(self, incoming, axis=1, **kwargs):
        super(Maxpooling, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.axis] + input_shape[(self.axis+1):]

    def get_output_for(self, input, **kwargs):
        return T.max(input, axis=self.axis)
