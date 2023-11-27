#creating the test, train and eval datasets
import sys
import random

file_name = sys.argv[1]

#read in all data from human_heavy.txt
data = []
with open(file_name, 'r') as f:
    for line in f:
        data.append(line)
        
random.shuffle(data)

#split the data into train, eval and test sets
#train_data = data[:int(len(data)*0.8)]
#eval_data = data[int(len(data)*0.8):int(len(data)*0.9)]
#test_data = data[int(len(data)*0.9):]

#train_data = data[:15000000]
#eval_data = data[15000000:16875000]
#test_data = data[16875000:18750000]

#write the train, eval and test sets to files
with open("train_large_heavy.txt", 'w') as f:
    for item in data[:15000000]:
        f.write("%s" % item)

with open("val_large_heavy.txt", 'w') as f:
    for item in data[15000000:16875000]:
        f.write("%s" % item)

with open("test_large_heavy.txt", 'w') as f:
    for item in data[16875000:18750000]:
        f.write("%s" % item)