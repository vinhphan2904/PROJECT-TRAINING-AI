from lib import *

source = '/Users/phanhuynhtuankhanh/Downloads/Downloads/Lettuce_disease_datasets/Healthy'
train_dir = './data/dataset/train'
val_dir = './data/dataset/val'
path = 'healthy'
import os

print("WORKING DIR:", os.getcwd())
print("TRAIN FOLDER:", os.path.abspath(os.path.join(train_dir, path)))
print("VAL FOLDER:", os.path.abspath(os.path.join(val_dir, path)))
