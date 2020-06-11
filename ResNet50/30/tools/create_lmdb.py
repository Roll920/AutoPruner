# loads imagenet and writes it into one massive binary file

import os
import numpy as np
from tensorpack.dataflow import *
import sys
if __name__ == '__main__':
    if len(sys.argv) < 4 :
        print("Usage: python create_lmdb.py gt_file.txt root_folder target_lmdb_name")
        print("gt_file.txt split by \"\t\"")
        sys.exit(1)
    class BinaryDataSet(RNGDataFlow):
        def __init__(self,text_name,root_folder):
            self.text_name = text_name
            self.length = 0
            self.root_folder = root_folder
            with open(self.text_name,'r') as f:
                self.length = len(f.readlines())
            self.gt_list = []
            with open(self.text_name,'r') as f:
                for line in f:

                    now_list = line.split('\t')
                    fname = now_list[0]
                    label = int(now_list[1].strip())
                    self.gt_list.append((fname,label))
        def size(self):
            return self.length
        def get_data(self):
            for fname, label in self.gt_list:
                with open(os.path.join(self.root_folder,fname), 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]
    gt_filename = sys.argv[1]
    root_folder = sys.argv[2]
    name = sys.argv[3]
    ds0 = BinaryDataSet(gt_filename,root_folder)
    ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
    dftools.dump_dataflow_to_lmdb(ds1, '%s.lmdb'%name)
