import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # Get shapes
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in xrange(num_train):
        # Compute vector of scores
        # f = np.array(X[i].dot(W))
        f = np.dot(X[i], W)

        # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
        f -= np.max(f)

        exp_f = np.exp(f)
        correct_class_score = exp_f[y[i]]
        # print correct_class_score
        loss += -np.log(correct_class_score / np.sum(exp_f))

        # [methods 1]: use exp style
        # count = 0
        # for j in xrange(num_classes):
        #   if j == y[i]:
        #     continue
        #   else:
        #     dW[:, j] += X[i] * scores[j] / np.sum(scores)
        #     count += scores[j]
        # dW[:, y[i]] += -count * X[i] / np.sum(scores)

        # [methods 2]: use probability style
        for j in range(num_classes):
            p = exp_f[j] / np.sum(exp_f)
            dW[:, j] += (p - (j == y[i])) * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # Get shapes
    num_train = X.shape[0]

    # Compute scores
    f = np.dot(X, W)

    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    f -= np.max(f)

    # Compute vector of stacked correct f-scores: [f(x_1)_{y_1}, ..., f(x_N)_{y_N}]
    exp_f = np.exp(f)
    correct_class_score = exp_f[np.arange(num_train), y]

    # loss = -sum all the minibatch( log(exp(correct_score) / demoninator))) / num_train
    demoninator = np.sum(exp_f, axis=1)
    loss = -np.mean(np.log(correct_class_score / demoninator))

    # [methods 1]: use exp style
    # mat = scores.T
    # counts = demoninator - correct_class_score  # counts[500,]
    # mat[y, range(num_train)] = -counts[range(num_train)]  # mat[10*500]
    # mat /= demoninator
    # dW = (np.dot(mat, X)).T

    # [methods 2]: use probability style
    p = exp_f / np.sum(exp_f, axis=1, keepdims = True)
    p[range(num_train), y] -= 1
    dW = np.dot(X.T, p)
    
    # mean
    dW /= num_train

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W # dW[3073*10]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW

