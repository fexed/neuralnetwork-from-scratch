import csv
import random
import math


cupfile = open("datasets/CUP/ML-CUP21-TR.csv")
testfile = open("datasets/CUP/internal_test_set_CUP.csv", "w", newline='')
trainingfile = open("datasets/CUP/training_set_CUP.csv", "w", newline='')
reader = csv.reader(cupfile)
writer_test = csv.writer(testfile)
writer_training = csv.writer(trainingfile)
total_lines = 1484
test_indexes = random.sample(range(7, total_lines), math.floor((total_lines-7)*0.1))
idx = 0
for row in reader:
    if "#" in row[0]:
        writer_test.writerow(row)
        writer_training.writerow(row)
        continue
    if idx in test_indexes:
        writer_test.writerow(row)
    else:
        writer_training.writerow(row)
    idx += 1
cupfile.close()
testfile.close()
trainingfile.close()