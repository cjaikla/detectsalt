import os
import re
import json

file_names=os.listdir(r"I:\Krongrath\Transfer\dataset\imageFile\train_threat\train_threat_augment")

list_train_threat_augment = []
for name in file_names:
	start = name.find('ID_') + 3
	end = name.find('_angle', start)
	only_name = name[start:end]
	list_train_threat_augment.append(only_name)


list_train_threat_augment = list(set(list_train_threat_augment))
print(len(list_train_threat_augment))

with open(r'.\dataset\train_threat_augment.txt', 'w') as outfile:
    json.dump(list_train_threat_augment, outfile)

with open(r'.\dataset\train_threat_augment.txt', 'r') as f:
    IDlist = json.loads(f.read())

print(len(IDlist))