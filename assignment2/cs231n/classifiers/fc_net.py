import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    # W1_flatten = np.random.normal(0, weight_scale, input_dim * hidden_dim)
    # self.params['W1'] = np.reshape(W1_flatten, (input_dim, hidden_dim))
    # W2_flatten = np.random.normal(0, weight_scale, hidden_dim * num_classes)
    # self.params['W2'] = np.reshape(W2_flatten, (hidden_dim, num_classes))
    
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    # scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    N = X.shape[0]
    X_reshape = np.reshape(X, (N,-1))
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X_reshape.shape
    # evaluate class scores with a 2-layer neural network
    hidden_layer = np.maximum(0, np.dot(X_reshape, W1) + b1)  # note, relu activation
    scores = np.dot(hidden_layer, W2) + b2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    reg = self.reg
    
    # compute the loss: average cross-entropy loss and regularization loss
    # average cross-entropy loss
    data_loss, dscores = softmax_loss(scores, y)
    # regularization loss
    reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    # add
    loss = data_loss + reg_loss
    
    
    # backpropate the gradient to the parameters
    # first backprop into parameters w2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the relu non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into w,b
    dW1 = np.dot(X_reshape.T, dhidden)
    db1 = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1

    # write into dict
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    dims = [input_dim] + hidden_dims + [num_classes]
    for i in xrange(self.num_layers):
      self.params['b%d' % (i+1)] = np.zeros(dims[i + 1])
      self.params['W%d' % (i+1)] = np.random.normal( 0, weight_scale, [dims[i], dims[i + 1]])
      if self.use_batchnorm:
          self.params['G%d' % (i+1)] = np.random.normal( 1 , 1e-3, dims[i + 1] )
          self.params['B%d' % (i+1)] = np.zeros(dims[i + 1])
            
    # last layer do not use batch normalization
    if self.use_batchnorm:
        del self.params['G%d' % self.num_layers]
        del self.params['B%d' % self.num_layers]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)



  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """            
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    # scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    layer_output = {}
    layer_cache = {}
    
    # evaluate till the last hidden layers
    layer_output[0] = X
    for i in range(1, self.num_layers):
        if self.use_batchnorm:
            layer_output[i], layer_cache[i] = affine_bn_relu_forward(
                                               layer_output[i-1],
                                               self.params['W%d' % (i)], 
                                               self.params['b%d' % (i)],
                                               self.params['G%d' % (i)],
                                               self.params['B%d' % (i)],
                                               self.bn_params[i-1])
            fc_cache, bn_cache, relu_cache = layer_cache[i]
   
        else:
            layer_output[i], layer_cache[i] = affine_relu_forward(layer_output[i-1],
                                                           self.params['W%d' % (i)], 
                                                           self.params['b%d' % (i)])
        if self.use_dropout:
            layer_output[i], drop_cache = dropout_forward(layer_output[i], self.dropout_param)
            layer_cache['dpcache%d'%(i)] = drop_cache
            
    W_last = 'W'+str(self.num_layers)
    b_last = 'b'+str(self.num_layers)
    
    # evaluate the output layer
    scores, output_cache = affine_forward(layer_output[self.num_layers-1],
                                          self.params[W_last],
                                          self.params[b_last])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ###########################################################################
    # compute the loss: average cross-entropy loss and regularization loss
    # average cross-entropy loss
    data_loss, dscores = softmax_loss(scores, y)

    # regularization loss
    reg_loss = 0
    for i in range(1, self.num_layers+1):
        w = self.params['W%d' % (i)]
        reg_loss += 0.5 * self.reg * np.sum(w * w)
        
    loss = data_loss + reg_loss
    #############################################################################
    # todo: compute the backward pass, computing the derivatives of the weights #
    # and biases. store the results in the grads dictionary. for example,       #
    # grads['w1'] should store the gradient on w1, and be a matrix of same size #
    #############################################################################
    # backpropate the gradient to the parameters
    grads = {}
    reg = self.reg
    
    # Backprop the last affine layer
    dx = {}
    dx[self.num_layers],grads[W_last],grads[b_last] = affine_backward(dscores, output_cache)
    grads[W_last] += reg * self.params[W_last]
    
    # next backprop into L-1 hidden layer
    for i in reversed(range(1, self.num_layers)):
        if self.use_dropout:
            dx[i+1] = dropout_backward(dx[i+1], layer_cache['dpcache%d'%(i)])
            
        if self.use_batchnorm:
            dx[i], grads['W%d' % i], grads['b%d' % i], grads['G%d' % i], grads['B%d' % i] = \
            affine_bn_relu_backward(dx[i+1], layer_cache[i])
        else:
            dx[i], grads['W%d' % i], grads['b%d' % i] = \
            affine_relu_backward(dx[i + 1], layer_cache[i])
      
        # add regularization gradient contribution
        grads['W%d' % i] += self.reg * self.params['W%d' % i]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return loss, grads
