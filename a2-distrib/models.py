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


class DAN(nn.Module):
    """
    Deep averaging network (DAN) to be trained for sentiment classification task.

    Modified version of functions from ffnn_example.py (CS 371N PyTorch tutorial).
    """
    def __init__(self, word_embed: WordEmbeddings, inp: int, hid1: int, hid2: int, out: int):
        """
        Constructs the DAN by instantiating the various layers and initializing weights.

        :param word_embed: word embeddings for given vocabulary.
        :param inp: size of input (integer).
        :param hid1: size of hidden layer 1 (integer).
        :param hid2: size of hidden layer 2 (integer).
        :param out: size of output (integer), which should be the number of classes.
        """
        super(DAN, self).__init__()
        self.word_embed = word_embed.get_initialized_embedding_layer(frozen=True)
        self.linear_1 = nn.Linear(inp, hid1)
        self.linear_2 = nn.Linear(hid1, hid2)
        self.linear_3 = nn.Linear(hid2, out)
        self.log_softmax = nn.LogSoftmax(dim=1) # Log probability distribution.
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """ 
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data.
        :return: an [out]-sized tensor of log probabilities.
        """
        res = self.word_embed(x)
        res = torch.mean(res, dim=1, keepdim=False) # Average word embeddings.
        res = self.linear_1(res)
        res = self.relu(res)
        res = self.linear_2(res)
        res = self.relu(res)
        res = self.linear_3(res)
        res = self.log_softmax(res)
        return res


def form_input(x) -> torch.Tensor:
    """ TODO: fix
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data.
    :return: a [num_samples x inp] Tensor.

    Modified version of form_input from ffnn_example.py (CS 371N PyTorch tutorial).
    """
    return torch.from_numpy(x).long()


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Trained deep averaging network (DAN) classifier and word embeddings
    to be used for sentiment classification.
    """
    def __init__(self, model: DAN, word_embed: WordEmbeddings):
        """ 
        :param model: trained DAN.
        :word_embed: word embedding for given vocabulary.
        """
        self.model = model
        self.word_embed = word_embed
    
    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence.
        :param ex_words: words to predict on.
        :return: 0 or 1 with the label.
        """
        # Get word embedding for each word in given sentence and format into tensor.
        sentence = [ex_words]
        x = process_x(sentence, self.word_embed, len(ex_words))
        x = form_input(np.array(x))
        
        # Predict sentence sentiment based on log probabilities.
        log_probs = self.model.forward(x)
        prediction = torch.argmax(log_probs)
        return prediction


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """ 
    :param args: Command-line args so you can access them here,
    :param train_exs: training examples.
    :param dev_exs: development set, in case you wish to evaluate your model during training.
    :param word_embeddings: set of loaded word embeddings.
    :return: A trained NeuralSentimentClassifier model.

    Modified version of main from ffnn_example.py (CS 371N PyTorch tutorial).
    """
    # DAN input size is the length of a word embedding vector.
    feat_vec_size = word_embeddings.get_embedding_length()

    # Hidden unit input sizes.
    hidden_1 = 100
    hidden_2 = 50

    # Output size of DAN (binary classification).
    num_classes = 2

    # Train and test DAN.
    num_epochs = 10
    batch_size = 1
    dan = DAN(word_embeddings, feat_vec_size, hidden_1, hidden_2, num_classes)
    initial_learning_rate = 0.01
    optimizer = optim.Adam(dan.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        # Shuffle indices of training examples for each epoch.
        total_loss = 0.0
        ex_indices = [i for i in range(0, len(train_exs))]
        random.seed(27)
        random.shuffle(ex_indices)

        # Batched neural net updates.
        start_idx = 0
        end_idx = start_idx + batch_size
        while start_idx < len(ex_indices):
            # Get word embedding for each word and place into tensor.
            batch_idxs = ex_indices[start_idx : end_idx]
            sentences = [train_exs[i].words for i in batch_idxs]
            max_len_sentence = len(max(sentences, key=len))

            # Add padding to sentences in batch.
            batch_x = process_x(sentences, word_embeddings, max_len_sentence)
            x = form_input(np.array(batch_x))

            # Build one-hot representation of y. 
            labels = [train_exs[i].label for i in batch_idxs]
            y_onehot = [torch.zeros(num_classes).scatter_(0, torch.from_numpy(np.asarray(labels[i],dtype=np.int64)), 1) for i in range(len(labels))]
            batch_y = np.array([list(y_onehot[i]) for i in range(len(y_onehot))])
            y = form_input(batch_y)

            # Zero out the gradients from the DAN object and get log probability for each word and class.
            dan.zero_grad()
            log_probs = dan.forward(x)

            # Calculate loss with negative log likelihood.
            loss = torch.sum(torch.neg(log_probs) * y)
            total_loss += loss

            # Computes the gradient and takes the optimizer step.
            loss.backward()
            optimizer.step()

            # Move index for start of batch.
            start_idx += batch_size
            end_idx += batch_size
        print("Total loss on epoch %i: %f" % (epoch, total_loss)) 

        # Evaluate on the dev set.
        dev_correct = 0
        for idx in range(0, len(dev_exs)):
            # Format input into tensor.
            sentence = [dev_exs[idx].words]
            x = process_x(sentence, word_embeddings, len(dev_exs[idx].words))
            x = form_input(np.array(x))
            y = dev_exs[idx].label
            log_probs = dan.forward(x)
            prediction = torch.argmax(log_probs)
            if y == prediction:
                dev_correct += 1
        print(repr(dev_correct / len(dev_exs)) + " dev accuracy\n")

    # Evaluate on the train set.
    train_correct = 0
    for idx in range(0, len(train_exs)):
        sentence = [train_exs[idx].words]
        x = process_x(sentence, word_embeddings, len(train_exs[idx].words))
        x = form_input(np.array(x))
        y = train_exs[idx].label
        log_probs = dan.forward(x)
        prediction = torch.argmax(log_probs)
        if y == prediction:
            train_correct += 1
    print(repr(train_correct) + "/" + repr(len(train_exs)) + " correct after training\n")

    return NeuralSentimentClassifier(dan, word_embeddings)

def process_x(sentences, word_embeddings: WordEmbeddings, max_len_sentence: int) -> list():
    """
    Gets index for each word in sentence.
    """
    batch_x = list()
    for sen in sentences:
        embed = [word_embeddings.word_indexer.index_of(sen[i]) for i in range(len(sen))]
        for i in range(len(embed)):
            if embed[i] == -1:
                embed[i] = word_embeddings.word_indexer.index_of("UNK")
        while len(embed) < max_len_sentence:
            embed.append(word_embeddings.word_indexer.index_of("PAD"))

        # Add word embeddings for sentence into batch.
        batch_x.append(embed)
    return batch_x


