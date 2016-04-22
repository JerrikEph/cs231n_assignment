import numpy as np
from random import shuffle

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
  train_nums = X.shape[0]
  class_nums = W.shape[1]

  for i in range(train_nums):
    scores = np.dot(X[i], W)
    sub_scores = scores - np.max(scores)
    exp_scores = np.exp(sub_scores)

    loss += np.log(np.sum(exp_scores)/exp_scores[y[i]])

    d_scores = exp_scores/np.sum(exp_scores)
    d_scores[y[i]] = exp_scores[y[i]]/np.sum(exp_scores) - 1  # C,
    dW += np.dot(X[i].reshape(-1, 1), d_scores.reshape(1, -1))

  loss /= train_nums
  dW /= train_nums

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
  train_nums = X.shape[0]
  class_nums = W.shape[1]

  scores = np.dot(X, W)   ## N, C
  sub_scores = scores - np.max(scores, 1).reshape(-1, 1)
  exp_scores = np.exp(sub_scores)   ## N, C

  loss = np.sum(np.log(np.sum(exp_scores, axis=1)) - np.log(exp_scores[np.arange(train_nums), y]))/train_nums + 0.5 * reg * np.sum(W * W) ##Compute loss

  d_scrores = exp_scores/np.sum(exp_scores, axis=1).reshape(-1, 1)       ## N, C
  d_scrores[np.arange(train_nums), y] = exp_scores[np.arange(train_nums), y]/np.sum(exp_scores, axis=1) -1
  dW = np.sum(X.reshape(X.shape[0], X.shape[1], 1) * d_scrores.reshape(d_scrores.shape[0], 1, d_scrores.shape[1]),
              axis=0)/train_nums + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

