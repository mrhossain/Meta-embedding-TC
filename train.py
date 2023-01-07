from __future__ import division
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import yaml
import math
import fasttext
import sys
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim

import re

import fasttext


from numpy import array
from sklearn.decomposition import TruncatedSVD

#import matplotlib.pyplot as plt

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.001, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim",300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters",128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size",128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs",30, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("decay_coefficient", 2.5, "Decay coefficient (default: 2.5)1.5")

FLAGS = tf.app.flags.FLAGS
start_time = time.time()


def load_average_embedding_vectors_glove_fastext(vocabulary, filenameg, filenamef, vector_size):
    print(len(vocabulary), vector_size, filenamef)

    model = fasttext.load_model(filenamef)
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))

    # load embedding_vectors from the glove
    # initial matrix with random uniform
    #model = FastText.load(filenamef)
    # model = fasttext.load_model(filenamef)
    # embedding_vectorsf = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    # for word in model.wv.vocab:
    #     vector = np.asarray(model[word], dtype="float32")
    #     idx = vocabulary.get(word)
    #     if idx != 0:
    #         embedding_vectorsf[idx] = vector

    # wordlist = model.words
    # for i in range(1, len(wordlist)):
    #     word = wordlist[i]
    #     # print("word = ",word)
    #     vector = np.asarray(model[wordlist[i]], dtype="float32")
    #     idx = vocabulary.get(word)
    #     if idx != 0:
    #         embedding_vectorsf[idx] = vector

    print(len(vocabulary), vector_size, filenameg)
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectorsg = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filenameg, encoding='utf-8', errors='ignore')
    for line in f:
        values = line.split()
        word = values[0]
        # print("word = ",word)
        vectorg = np.asarray(values[1:], dtype="float32")
        vectorf = np.asarray(model[word], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = np.mean(np.array([vectorg, vectorf]), axis=0)
            print("Len = ",len(embedding_vectors[idx]))
    f.close()
    #embedding_vectors = np.mean(np.array([embedding_vectorsg, embedding_vectorsf]), axis=0)
    return embedding_vectors


def load_concate_embedding_vectors_glove_fastext(vocabulary, filenameg, filenamef, vector_size):
    print(len(vocabulary), vector_size, filenamef)
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    #model = FastText.load(filenamef)

    model = fasttext.load_model(filenamef)
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    # for word in model.wv.vocab:
    #     vector = np.asarray(model[word], dtype="float32")
    #     idx = vocabulary.get(word)
    #     if idx != 0:
    #         embedding_vectorsf[idx] = vector

    #wordlist = model.words
    # for i in range(1, len(wordlist)):
    #     word = wordlist[i]
    #     # print("word = ",word)
    #     vector = np.asarray(model[wordlist[i]], dtype="float32")
    #     idx = vocabulary.get(word)
    #     if idx != 0:
    #         embedding_vectorsf[idx] = vector

    # embedding_vectorsf = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size//2))
    # for word in model.wv.vocab:
    #     vector = np.asarray(model[word], dtype="float32")
    #     idx = vocabulary.get(word)
    #     if idx != 0:
    #         embedding_vectorsf[idx] = vector
    #
    # print(len(vocabulary), vector_size, filenameg)
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    #embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filenameg,encoding='utf-8', errors='ignore')
    for line in f:
        values = line.split()
        word = values[0]
        # print("word = ",word)
        vectorg = np.asarray(values[1:], dtype="float32")
        vectorf = np.asarray(model[word], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            # A = array([vectorg,vectorf])
            # svd = TruncatedSVD(n_components=150)
            # svd.fit(A)
            # result = svd.transform(A)
            # print(result)
            # gg = np.asarray(result[0], dtype="float32")
            # ff = np.asarray(result[1], dtype="float32")
            # gg = gg[1:]
            # ff = ff[1:]
            embedding_vectors[idx] = np.concatenate(np.array([vectorg, vectorf]), axis=0)
    f.close()
    return embedding_vectors



def load_embedding_vectors_fastext_gensim(vocabulary,filenamef, vector_size):
    print(len(vocabulary), vector_size, filenamef)
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    model = FastText.load(filenamef)
    embedding_vectorsf = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    for word in model.wv.vocab:

        if word in model.wv.vocab:
            vector = np.asarray(model[word], dtype="float32")
            idx = vocabulary.get(word)
            if idx != 0:
                embedding_vectorsf[idx] = vector

    return embedding_vectorsf


def load_embedding_vectors_word2vec_gensim(vocabulary,filenamef, vector_size):
    print(len(vocabulary), vector_size, filenamef)
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    model = gensim.models.Word2Vec.load(filenamef)
    embedding_vectorsf = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    for word in model.wv.vocab:
        if word in model.wv.vocab:
            vector = np.asarray(model[word], dtype="float32")
            idx = vocabulary.get(word)
            if idx != 0:
                embedding_vectorsf[idx] = vector

    return embedding_vectorsf


def preprocess(text):
    text = str(text).replace('।', '\n')
    whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
    bangla_fullstop = u"\u0964"
    punctSeq = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
    punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"
    text = whitespace.sub(" ", text).strip()
    text = re.sub(punctSeq, " ", text)
    text = re.sub(punc, " ", text)
    text = "".join(i for i in text if ord(i) > ord('z') or ord(i) == 32)
    text = re.sub(' +', ' ', text)
    return (text)


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))


with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


dataset_name = cfg["datasets"]["default"]

print("dataset_name",dataset_name)

if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
else:
    embedding_dimension = FLAGS.embedding_dim




container_path = cfg["datasets"][dataset_name]["container_path"]
categories = cfg["datasets"][dataset_name]["categories"]
shuffle = cfg["datasets"][dataset_name]["shuffle"]
random_state = cfg["datasets"][dataset_name]["random_state"]

    #datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],
     #                                                categories=cfg["datasets"][dataset_name]["categories"],
      #                                               shuffle=cfg["datasets"][dataset_name]["shuffle"],
       #                                              random_state=cfg["datasets"][dataset_name]["random_state"])


datasets = load_files(container_path=container_path, categories=None, load_content=True,shuffle=shuffle, encoding='utf-8', random_state=random_state)

x_text = datasets['data']

# for i in range(len(x_text)):
#     x_text[i] = preprocess(x_text[i])

total_documents = len(x_text)
x_text = [document.strip() for document in x_text]
labels = []
total_documents
max_document_length = 0
for i in range(len(x_text)):
    #x_text[i] = preprocess(x_text[i])
    label = [0 for j in datasets['target_names']]
    label[datasets['target'][i]] = 1
    labels.append(label)
    max_document_length = max(max_document_length,len(x_text[i].split()))


print("max_document_length.......... = ",max_document_length)
max_document_length = 256


y = np.array(labels)
print("X_text_len",len(x_text))

# Build vocabulary
#print(".......................................")
#max_document_length = max([len(x.split()) for x in x_text])

max_document_length_wrong = max([len(x.split(" ")) for x in x_text])


print("Max Document Length1 = ",max_document_length)
print("max_document_length_wrong",max_document_length_wrong)


#print("...............end........................")
print("...Train max len.. = ",len(x_text))

print(".......................step3................")
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
print("max_document_length = ",max_document_length)
print(".......................step4................")
#print(vocab_processor)
x = np.array(list(vocab_processor.fit_transform(x_text)))

print(".......................step5................")
print(x)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
del x, y, x_shuffled, y_shuffled
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

print("--- %s seconds ---" % (time.time() - start_time))

glove_embeddng_path = '/mnt/b7e83844-6bc7-43cd-8eee-87edc0eadf1e/glove/glove/Covid-Embedding/vectors.txt'
fastetxt_embedding_path = '/mnt/b7e83844-6bc7-43cd-8eee-87edc0eadf1e/FastText/fastText/Embedding-News-Corpus-200-300/News-embedding300.bin'

#print("............train...........",x_train)
#print(".....................dev....",x_dev)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dimension,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(cnn.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        print(".......................step6................")
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = vocab_processor.vocabulary_
            initW = None
            if embedding_name == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initW = data_helpers.load_embedding_vectors_word2vec(vocabulary,cfg['word_embeddings']['word2vec']['path'],cfg['word_embeddings']['word2vec']['binary'])
                print("word2vec...file has been loaded")
            elif embedding_name == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                initW = data_helpers.load_embedding_vectors_glove(vocabulary,cfg['word_embeddings']['glove']['path'],embedding_dimension)
                print("glove file has been loaded\n")
            elif embedding_name == 'fasttext':
                # load embedding vectors from the glove
                print("Load fasttext file {}".format(cfg['word_embeddings']['fasttext']['path']))
                initW = data_helpers.load_embedding_vectors_fasttext(vocabulary, cfg['word_embeddings']['fasttext']['path'],embedding_dimension)
                print("fasttext file has been loaded\n")
            elif embedding_name == 'average_g_f':
                initW = load_average_embedding_vectors_glove_fastext(vocabulary,glove_embeddng_path,fastetxt_embedding_path,embedding_dimension)
                print("fasttext and GloVe file has been loaded\n")

            elif embedding_name == 'concate_g_f':
                initW = load_concate_embedding_vectors_glove_fastext(vocabulary,glove_embeddng_path,fastetxt_embedding_path,embedding_dimension)
                print("fasttext and GloVe file has been loaded\n")

            elif embedding_name == 'fasttext-gensim':
                initW = load_embedding_vectors_fastext_gensim(vocabulary,cfg['word_embeddings']['fasttext-gensim']['path'],embedding_dimension)
                print("fasttext has been loaded\n")

            elif embedding_name == 'word2vec-gensim':
                initW = load_embedding_vectors_fastext_gensim(vocabulary,cfg['word_embeddings']['word2vec-gensim']['path'],embedding_dimension)
                print("word2vec has been loaded\n")


            sess.run(cnn.W.assign(initW))

        def train_step(x_batch, y_batch, learning_rate):
            """
            A single training step
            """
            #print("...........x_batch,y_batch..............",x_batch,y_batch)
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
              cnn.learning_rate: learning_rate
            }
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            sloss = (str)(step) + " " + (str)(loss)
            sacc = (str)(step) + " "+ (str)(accuracy)
            with open("train_loss.txt", "a") as myfile:
                myfile.write(sloss)
                myfile.write('\n')
            with open("train_acc.txt", "a") as myfile1:
                myfile1.write(sacc)
                myfile1.write('\n')

            print("{}: step {}, loss {:g}, acc {:g}, learning_rate {:g}"
                  .format(time_str, step, loss, accuracy, learning_rate))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            print("........................x_batch len............... = ",len(x_batch),len(x_batch[0]) )
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()

            sloss = (str)(step) + " " + (str)(loss)
            sacc = (str)(step) + " " + (str)(accuracy)
            with open("Dev_loss.txt", "a") as myfile:
                myfile.write(sloss)
                myfile.write('\n')
            with open("Dev_acc.txt", "a") as myfile1:
                myfile1.write(sacc)
                myfile1.write('\n')

            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        print(".......................step7................")
        batches = data_helpers.batch_iter( list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        print("......batches.......",batches)
        print(".......................step8................")
        # It uses dynamic learning rate with a high value at the beginning to speed up the training
        #print(y_train)
        max_learning_rate = 0.005
        min_learning_rate = 0.0001
        #print(FLAGS.decay_coefficient)
        #print(len(y_train))
       # print(FLAGS.batch_size)
        decay_speed = FLAGS.decay_coefficient*len(y_train)/FLAGS.batch_size
        print("decay_speed =",decay_speed)
        # Training loop. For each batch...
        counter = 0
        #print(batches)
        epoch_sum = 0
        #epoch = 0
        #lr = 0.01
        # modified my me
        #learning_rate = lr #min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter / decay_speed)
        for batch in batches:
            current_step = tf.train.global_step(sess, global_step)
            epoch_sum =  epoch_sum + FLAGS.batch_size

            # modified my me
            #if epoch_sum >= total_documents:
             #   epoch_sum = 0
              #  epoch = epoch+1
               # print("Learning Rate  Epoch = ",learning_rate,epoch)
                #if epoch % 10 ==0:
                 #   learning_rate = learning_rate*0.1 #min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch / decay_speed)

            #print("current_step = ",current_step)
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
            counter += 1
            #print("................",counter)
            x_batch, y_batch = zip(*batch)
            print("..x_batch...........",len(x_batch))
            train_step(x_batch, y_batch, learning_rate)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
