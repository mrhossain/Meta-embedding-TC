import os
import csv
basepath = '/mnt/58a96951-a863-483e-b6bd-01c265d94667/single-layer-multikernel-cnn/Marathi-cnn-model/data'
out_file =  open('CUETNLP_subtask-1f-test_run-5.txt', 'a+')

def get_label(label):
    ss = label
    label= str(ss).strip('[]')
    label = label.replace("'","")
    return label

csv_files = []
for file in os.listdir(basepath):
    print(file)
    path = os.path.join(basepath,file)
    p = open(path,'r')
    csv_files.append(csv.reader(p, delimiter='\t'))


idx = 0
running = True
while running:
    labels = {}
    for csv_file in csv_files:
        try:
            row = next(csv_file)
            label = get_label(row[1])
            if label in labels:
                labels[label] += 1
            else:
                labels[label] = 1
        except:
            running = False
    
    if running:
        mx_key = list(labels.keys())[0]
        for key in labels.keys():
            if labels[key] > labels[mx_key]:
                mx_key = key
        print("%s\t%s" % (row[0], mx_key), file=out_file)
