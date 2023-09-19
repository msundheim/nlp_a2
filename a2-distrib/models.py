# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """ TODO: fix param lists
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, model, word_embed):
        self.model = model
        self.word_embed = word_embed
    
    def predict(self, ex_words: List[str]) -> int:
        word_sum = 0
        x = form_input(np.array([self.word_embed.get_embedding(ex_words[i]) for i in range(len(ex_words))]))
        log_probs = self.model.forward(x)
        prediction = torch.argmax(log_probs)
        return prediction


class DAN(nn.Module):
    """ TODO: fix comments and cite
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, word_embed, inp, hid1, hid2, out):
        """ TODO: complete and comments
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(DAN, self).__init__()

        # Get PyTorch embedding layer for network input.
        self.embeddings = word_embed.get_initialized_embedding_layer()
        self.linear_1 = nn.Linear(inp, hid1)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hid1, hid2)
        self.linear_3 = nn.Linear(hid2, out)
        self.log_softmax = nn.LogSoftmax(dim=0) # TODO: may need to fix dimension

    
    def forward(self, x):
        """ TODO: complete and comments
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        res = torch.mean(x, dim=0)
        res = self.linear_1(res)
        res = self.relu(res)
        res = self.linear_2(res)
        res = self.relu(res)
        res = self.linear_3(res)
        res = self.log_softmax(res)
        return res

    
def form_input(x) -> torch.Tensor:
    """ TODO: fix and cite
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(x).float()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """ TODO: fix and cite
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Inputs are of size 2
    feat_vec_size = word_embeddings.get_embedding_length()
    # Let's use 4 hidden units
    hidden_1 = 100
    hidden_2 = 50
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # RUN TRAINING AND TEST
    num_epochs = 5
    dan = DAN(word_embeddings, feat_vec_size, hidden_1, hidden_2, num_classes)
    initial_learning_rate = 0.01
    optimizer = optim.Adam(dan.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.seed(27)
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = form_input(np.array([word_embeddings.get_embedding(train_exs[idx].words[i]) for i in range(len(train_exs[idx].words))]))
            y = train_exs[idx].label
            # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
            # way we can take the dot product directly with a probability vector to get class probabilities.
            y_onehot = torch.zeros(num_classes)
            # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            dan.zero_grad()
            log_probs = dan.forward(x)
            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = torch.neg(log_probs).dot(y_onehot)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    # Evaluate on the train set
    train_correct = 0
    for idx in range(0, len(train_exs)):
        x = form_input(np.array([word_embeddings.get_embedding(train_exs[idx].words[i]) for i in range(len(train_exs[idx].words))]))
        y = train_exs[idx].label
        log_probs = dan.forward(x)
        prediction = torch.argmax(log_probs)
        if y == prediction:
            train_correct += 1
    print(repr(train_correct) + "/" + repr(len(train_exs)) + " correct after training\n")

    return NeuralSentimentClassifier(dan, word_embeddings)

