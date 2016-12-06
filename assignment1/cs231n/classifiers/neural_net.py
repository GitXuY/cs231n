import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  a two-layer fully-connected neural network. the net has an input dimension of
  n, a hidden layer dimension of h, and performs classification over c classes.
  we train the network with a softmax loss function and l2 regularization on the
  weight matrices. the network uses a relu nonlinearity after the first fully
  connected layer.

  in other words, the network has the following architecture:

  input - fully connected layer - relu - fully connected layer - softmax

  the outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    initialize the model. weights are initialized to small random values and
    biases are initialized to zero. weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    w1: first layer weights; has shape (d, h)
    b1: first layer biases; has shape (h,)
    w2: second layer weights; has shape (h, c)
    b2: second layer biases; has shape (c,)

    inputs:
    - input_size: the dimension d of the input data.
    - hidden_size: the number of neurons h in the hidden layer.
    - output_size: the number of classes c.
    """
    self.params = {}
    self.params['w1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['w2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, x, y=None, reg=0.0):
    """
    compute the loss and gradients for a two layer fully connected neural
    network.

    inputs:
    - x: input data of shape (n, d). each x[i] is a training sample.
    - y: vector of training labels. y[i] is the label for x[i], and each y[i] is
      an integer in the range 0 <= y[i] < c. this parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: regularization strength.

    returns:
    if y is None, return a matrix scores of shape (n, c) where scores[i, c] is
    the score for class c on input x[i].

    if y is not None, instead return a tuple of:
    - loss: loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # unpack variables from the params dictionary
    w1, b1 = self.params['w1'], self.params['b1']
    w2, b2 = self.params['w2'], self.params['b2']
    n, d = x.shape

    # compute the forward pass
    # scores = None
    #############################################################################
    # todo: perform the forward pass, computing the class scores for the input. #
    # store the result in the scores variable, which should be an array of      #
    # shape (n, c).                                                             #
    #############################################################################
    # evaluate class scores with a 2-layer neural network
    hidden_layer = np.maximum(0, np.dot(x, w1) + b1)  # note, relu activation
    scores = np.dot(hidden_layer, w2) + b2
    #############################################################################
    #                              end of your code                             #
    #############################################################################

    # if the targets are not given then jump out, we're done
    if y is None:
      return scores

    # compute the loss
    # loss = None
    #############################################################################
    # todo: finish the forward pass, and compute the loss. this should include  #
    # both the data loss and l2 regularization for w1 and w2. store the result  #
    # in the variable loss, which should be a scalar. use the softmax           #
    # classifier loss. so that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    # normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    scores -= np.max(scores)

    # compute the class probabilities
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # compute the loss: average cross-entropy loss and regularization
    # the log probabilities assigned to the correct classes in each example
    corect_logprobs = -np.log(probs[np.arange(n), y])
    data_loss = np.sum(corect_logprobs) / n
    reg_loss = 0.5 * reg * (np.sum(w1 * w1) + np.sum(w2 * w2))
    loss = data_loss + reg_loss
    #############################################################################
    #                              end of your code                             #
    #############################################################################

    # backward pass: compute gradients
    grads = {}
    #############################################################################
    # todo: compute the backward pass, computing the derivatives of the weights #
    # and biases. store the results in the grads dictionary. for example,       #
    # grads['w1'] should store the gradient on w1, and be a matrix of same size #
    #############################################################################
    # compute the gradient on scores
    dscores = probs
    dscores[range(n), y] -= 1
    dscores /= n

    # backpropate the gradient to the parameters
    # first backprop into parameters w2 and b2
    dw2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, w2.T)
    # backprop the relu non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into w,b
    dw1 = np.dot(x.T, dhidden)
    db1 = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    dw2 += reg * w2
    dw1 += reg * w1

    # write into dict
    grads['w1'] = dw1
    grads['w2'] = dw2
    grads['b1'] = db1
    grads['b2'] = db2
    #############################################################################
    #                              end of your code                             #
    #############################################################################

    return loss, grads

  def train(self, x, y, x_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    train this neural network using stochastic gradient descent.

    inputs:
    - x: a numpy array of shape (n, d) giving training data.
    - y: a numpy array f shape (n,) giving training labels; y[i] = c means that
      x[i] has label c, where 0 <= c < c.
    - x_val: a numpy array of shape (n_val, d) giving validation data.
    - y_val: a numpy array of shape (n_val,) giving validation labels.
    - learning_rate: scalar giving learning rate for optimization.
    - learning_rate_decay: scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: scalar giving regularization strength.
    - num_iters: number of steps to take when optimizing.
    - batch_size: number of training examples to use per step.
    - verbose: boolean; if True print progress during optimization.
    """
    num_train = x.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # use sgd to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      x_batch = None
      y_batch = None

      #########################################################################
      # todo: create a random minibatch of training data and labels, storing  #
      # them in x_batch and y_batch respectively.                             #
      #########################################################################
      random_index = np.random.choice(num_train, batch_size)
      x_batch = x[random_index]
      y_batch = y[random_index]
      #########################################################################
      #                       end of your code                                #
      #########################################################################
      #########################################################################
      #                             end of your code                          #
      #########################################################################

      # compute loss and gradients using the current minibatch
      loss, grads = self.loss(x_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # todo: use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. you'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['w1'] += - learning_rate * grads['w1']
      self.params['w2'] += - learning_rate * grads['w2']
      self.params['b1'] += - learning_rate * grads['b1']
      self.params['b2'] += - learning_rate * grads['b2']
      #########################################################################
      #                             end of your code                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # check accuracy
        train_acc = (self.predict(x_batch) == y_batch).mean()
        val_acc = (self.predict(x_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, x):
    """
    use the trained weights of this two-layer network to predict labels for
    data points. for each data point we predict scores for each of the c
    classes, and assign each data point to the class with the highest score.

    inputs:
    - x: a numpy array of shape (n, d) giving n d-dimensional data points to
      classify.

    returns:
    - y_pred: a numpy array of shape (n,) giving predicted labels for each of
      the elements of x. for all i, y_pred[i] = c means that x[i] is predicted
      to have class c, where 0 <= c < c.
    """
    # y_pred = None

    ###########################################################################
    # todo: implement this function; it should be very simple!                #
    ###########################################################################
    hidden_layer = np.maximum(0, np.dot(x, self.params['w1']) + self.params['b1'])  # note, relu activation
    scores = np.dot(hidden_layer, self.params['w2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              end of your code                           #
    ###########################################################################

    return y_pred


