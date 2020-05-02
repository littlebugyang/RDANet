import os
import shutil
import re
import filecmp

seq_dir = ''
n_frame = 7
avail_name = 'available_list.txt'

avail_path = os.path.join(os.getcwd(), seq_dir, avail_name)
temp_path = os.path.join(os.getcwd(), seq_dir, avail_name + '.temp')

avail_file = open(avail_path, mode='r')
temp_file = open(temp_path, mode='w')

avail_paths = avail_file.readlines()

for line in avail_paths:
    img_path = line.rstrip()
    half = int(n_frame / 2)
    split_result = re.split(r'[\\|/]', img_path)
    target_digit = int(split_result[-1][2:-4])

    offsets = [x for x in range(-half, half + n_frame % 2) if x != 0]

    duplicated = False
    for i in offsets:
        neigh_digit = target_digit + i
        neigh_name = 'im' + str(neigh_digit).zfill(5) + '.png'
        split_result[-1] = neigh_name
        neigh_path = '/'.join(split_result)

        if filecmp.cmp(img_path, neigh_path):
            duplicated = True
            print("Duplicated: ", img_path + ' and ' + neigh_path)
            break

    if not duplicated:
        temp_file.write(img_path + '\n')

avail_file.close()
temp_file.close()

# todo: delete origin avail_file and rename the temp_file
os.remove(avail_path)
os.rename(temp_path, avail_path)
