import shutil
import os

num_seq = 5000
num_digit = 7
seq_dir = 'sequences'

for i in range(num_seq):
    potential_dir = os.path.join(os.getcwd(), seq_dir, 'seq_'+str(i+1).zfill(num_digit))
    if os.path.exists(potential_dir):
        shutil.rmtree(potential_dir)
