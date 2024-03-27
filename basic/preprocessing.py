import os

file_path = './dataset'

train_path = os.path.join(file_path, 'train')
test_path = os.path.join(file_path, 'test')
# f = open("./dataset/train_file.txt", 'w')
# f2 = open("./dataset/train_file2.txt", 'w')
f3 = open("./dataset/train_data_list.txt", 'w')
f4 = open("./dataset/train_label_list.txt", 'w')

labels = []

for label in os.listdir(train_path):
    label_root = os.path.join(train_path, label)
    for image in os.listdir(label_root):
        item = [image, label]
        labels.append(item)

sort_labels = sorted(labels, key = lambda x:(x[1],x[0]))

for item in sort_labels:
    name, label = item
    f3.write(name+'\n')
    f4.write(label+'\n')
#     f2.write(name+' '+label+'\n')
#
# f2.close()


