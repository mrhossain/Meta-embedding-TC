import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        print("..vocab_size.......embedding_size....",vocab_size,embedding_size)
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            print("i.............filterSize.........",i,filter_size,embedding_size)
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                f1 = 3
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                #filter_shape = [filter_size, embedding_size - (f1 - 1), 1, 1]
                print("filter_shape = ",filter_shape)
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                print("W = ", W.shape)
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print("conv = ",conv.shape)
                print("self.embedded_chars_expanded",self.embedded_chars_expanded.shape)
                # Apply nonlinearity
                #h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h = tf.nn.leaky_relu(tf.nn.bias_add(conv, b), name="leaky_relu")
                #h = tf.nn.sigmoid(tf.nn.bias_add(conv, b), name="sigmoid")
                print("h ........= ",h)
                # Maxpooling over the outputs
                print("sequence_length = ",sequence_length)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print("pooled = ",pooled)
                pooled_outputs.append(pooled)
                print("pooled_outputs = ",pooled_outputs[i])

                print("leaky_reluy-h.shape",h.shape)
                print("pooled shape",pooled.shape)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        print("num_filters_total = ",num_filters_total)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total]) # -1 means flate the matrix to 1d array
        print("self.h_pool_flat = ", self.h_pool_flat)
        print("self.h_pool_flat_Len = ", self.h_pool_flat)
        print("h_pool = ",self.h_pool)
        print("self.h_pool_flat.shape = ", self.h_pool_flat.shape)
        print("h_poolshape = ", self.h_pool.shape)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            print("self.h_drop.shape",self.h_drop[1].shape)


        #Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",shape=[num_filters_total, num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            print("W = ",W.shape)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

