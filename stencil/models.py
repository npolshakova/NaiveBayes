#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   This file contains the Naive Bayes classifier

   Brown CS142, Spring 2019
"""
import random

import numpy as np


class NaiveBayes(object):
    """ Bernoulli Naive Bayes model

    @attrs:
        n_classes: the number of classes
    """

    def __init__(self, n_classes):
        """ Initializes a NaiveBayes classifer with n_classes. """
        self.n_classes = n_classes
        self.priors = np.ones(self.n_classes)
        self.prob = []
        self.n_labels = 0
        # You are free to add more fields here.

    def train(self, data):
        """ Trains the model, using maximum likelihood estimation.

        @params:
            data: the training data as a namedtuple with two fields: inputs and labels
        @return:
            None
        """
        self.n_labels = len(data[0][0])
        class_prior = np.ones(self.n_classes)
        attrib_dist = np.ones((self.n_classes, self.n_labels))
        for i in range(len(data[0])):
            inputs = data[0][i]
            label = data[1][i]
            class_prior[label] = class_prior[label] + 1
            for j in range(len(inputs)):
                attrib = inputs[j]
                if attrib == 1:
                    attrib_dist[label][j] = attrib_dist[label][j] + 1

        for i in range(len(attrib_dist)):
            attrib_dist[i] = np.true_divide(attrib_dist[i], class_prior[i] + 1)

        class_prior = np.true_divide(class_prior, len(data) + self.n_classes)

        self.priors = class_prior
        self.prob = attrib_dist
        print(self.accuracy(data))

    def predict(self, inputs):
        """ Outputs a predicted label for each input in inputs.

        @params:
            inputs: a NumPy array containing inputs
        @return:
            a numpy array of predictions
        """
        ret = []
        for i in range(len(inputs)):
            row = inputs[i]
            max_prob = 0.0
            max_class = 0.0
            for n in range(self.n_classes):
                    p = 1.0
                    for j in range(len(row)):
                        if row[j] == 1:
                            p = p * self.prob[n][j]
                        else:
                            p = p * (1.0 -  self.prob[n][j])
                    p = p * self.priors[n]
                    if p > max_prob:
                        max_prob = p
                        max_class = n
            ret.append(max_class)
        return ret


    def accuracy(self, data):
        """ Outputs the accuracy of the trained model on a given dataset (data).

        @params:
            data: a dataset to test the accuracy of the model.
            a namedtuple with two fields: inputs and labels
        @return:
            a float number indicating accuracy (between 0 and 1)
        """
        predictions = self.predict(data[0])
        return (predictions == data[1]).mean()
