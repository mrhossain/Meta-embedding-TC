from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import data_helpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml
import io
import sys
import re
from sklearn.datasets import load_files


def preprocess(text):
    punctSeq = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
    punc = u"[(),$%^&*+={}\[\]:\"|\'\~`<>/-]+"
    text = re.sub(punctSeq, " ", text)
    text = re.sub(punc, " ", text)
    text = re.sub(' +', ' ', text)
    return (text)



def softmax(x):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


# Parameters
# ==================================================
# Data Parameters


#/mnt/58a96951-a863-483e-b6bd-01c265d94667/single-layer-multikernel-cnn/runs/1659288437/checkpoints/model-1400
# Eval Parameters
tf.flags.DEFINE_integer("batch_size",128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", '/mnt/58a96951-a863-483e-b6bd-01c265d94667/single-layer-multikernel-cnn/runs/1659288437/checkpoints/', "")
tf.flags.DEFINE_boolean("eval_train",True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
datasets = None

#CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]
print("Local Data Loading...")

container_path = cfg["datasets"][dataset_name]["container_path"]
categories = cfg["datasets"][dataset_name]["categories"]
shuffle = cfg["datasets"][dataset_name]["shuffle"]
random_state = cfg["datasets"][dataset_name]["random_state"]




datasets = load_files(container_path=container_path, categories=None, load_content=True,shuffle=shuffle, encoding='utf-8', random_state=random_state)

x_raw = datasets['data']
#datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["container_path"],categories=cfg["datasets"][dataset_name]["categories"],shuffle=cfg["datasets"][dataset_name]["shuffle"],random_state=cfg["datasets"][dataset_name]["random_state"])

# x_raw, y_test = data_helpers.load_data_labels(datasets)
# actual_y = y_test
# y_test = np.argmax(y_test, axis=1)
# print(y_test)
# print("Total number of test examples: {}".format(len(y_test)))


total_documents = len(x_raw)
x_raw = [document.strip() for document in x_raw]
labels = []
total_documents
max_document_length = 0
for i in range(len(x_raw)):
    label = [0 for j in datasets['target_names']]
    label[datasets['target'][i]] = 1
    labels.append(label)
    max_document_length = max(max_document_length,len(x_raw[i].split()))


print("max_document_length.......... = ",max_document_length)
max_document_length = 256
#y = np.array(labels)
y_test = np.array(labels)
actual_y = y_test
y_test = np.argmax(y_test, axis=1)



# Map data into vocabulary

#
# for i in range( len(x_raw) ):
#     x_raw[i] = preprocess(x_raw[i])
#
#

vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================


#checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

print("Check File",checkpoint_file)

graph = tf.Graph()
with graph.as_default():

    session_conf = tf.ConfigProto( allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None
        for x_test_batch in batches:
            batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            #print(".............",len(probabilities))
            #print(probabilities)
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities


# Print accuracy if y_test is defined

label = datasets['target'][0]
print(label)

#label[datasets['target'][i]] = 1
#labels.append(label)
#print("Len",len(all_predictions))


basepath = '/mnt/58a96951-a863-483e-b6bd-01c265d94667/single-layer-multikernel-cnn/res'
print(all_probabilities.shape)
print(actual_y.shape)

'''

for i in range(0,750):
    a = int(y_test[i])
    b = int(all_predictions[i])
    if a!=b:
        print(a)
        #print(x_raw[i])
        filename = basepath+str(i)
        filename = filename + str(a)
        filename = filename+'.txt'
        print("Actual",y_test[i])
        print("Probability", all_probabilities[i])

        with io.open(filename, 'w', encoding='utf8') as f:
            f.write(str(y_test[i]))
            f.write(str(all_probabilities[i]))
            f.write(x_raw[i])
            f.close()

'''



#res  = ['bioche','com_tech','cse','mgmt','phy'] #Bengali
#res  = ['ai','algo','ca','cn','dbms','pro','se'] #CS
#res  = ['che','cse','law','math','physics'] #English
#res  = ['bioche','com_tech','cse','mgmt','phy'] #Gujarati
#res  = ['bioche','com_tech','cse','math','mgmt','other','phy'] #Hindi
#res  = ['bioche','com_tech','cse'] #Mala
#res  = ['bioche','com_tech','cse','phy'] #Mart
#res  = ['bioche','com_tech','cse','mgmt','other','phy'] #Tamil
#res  = ['bioche','com_tech','cse','mgmt','other','phy'] #Telugu
#print("Check",res[0])
#
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print(metrics.classification_report(y_test, all_predictions, target_names=datasets['target_names']))
    print(metrics.confusion_matrix(y_test, all_predictions))


# w_filenameTSV = '/mnt/1b47d000-2aee-4aa6-92a0-ff08c97e14fc/ICON-datasets-2020/OneDrive_1_10-30-2020/sub-task-1b/res.tsv'
# with open(w_filenameTSV,'w') as write_tsv:
#     write_tsv.write(csv_read.to_csv(sep='\t', index=False))

# out_file =  open('/mnt/58a96951-a863-483e-b6bd-01c265d94667/single-layer-multikernel-cnn/Marathi-cnn-model/data/CUETNLP_subtask-1f-test_run-4.txt', 'a+')
# for i in range( len(x_raw) ):
#     idx = int(all_predictions[i])
#     print("%s\t%s" % (x_raw[i], res[idx]), file=out_file)


# Save the evaluation to a csv
#predictions_human_readable = np.column_stack((np.array(x_raw),[int(prediction) for prediction in all_predictions],["{}".format(probability) for probability in all_probabilities]))

# predictions_human_readable = np.column_stack((np.array(x_raw),[int(prediction) for prediction in all_predictions],["{}".format(probability) for probability in all_probabilities]))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)


# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
#
# from sklearn import svm, datasets
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
# from sklearn.metrics import roc_auc_score
#
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
#
# n_classes = actual_y.shape[1]
# lw = 2
# fpr = dict()
# tpr = dict()
# threshhold = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], threshhold[i] = roc_curve(actual_y[:, i], all_probabilities[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#     #print(threshhold[i])
#
#
# fpr["micro"], tpr["micro"], _ = roc_curve(actual_y.ravel(), all_probabilities.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
# # Finally average it and compute AUC
# mean_tpr /= n_classes
#
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# category = ['Accident','Crime','Entertainment','Health','Politics','Sports']
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(category[i], roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()
#
#
#
#
# precision = dict()
# recall = dict()
# average_precision = dict()
# for i in range(n_classes):
#     precision[i], recall[i], _ = precision_recall_curve(actual_y[:, i], all_probabilities[:, i])
#     average_precision[i] = average_precision_score(actual_y[:, i], all_probabilities[:, i])
#
# # A "micro-average": quantifying score on all classes jointly
# precision["micro"], recall["micro"], _ = precision_recall_curve(actual_y.ravel(), all_probabilities.ravel())
# average_precision["micro"] = average_precision_score(actual_y, all_probabilities, average="micro")
# print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
#
# plt.figure()
# plt.step(recall['micro'], precision['micro'], where='post')
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
# plt.show()
#
#
#
#
