from lib import *
from dataset_create import MyDataset
from imagetransform import ImageTransform
from datalist import train_list,val_list
from normalize import resize,mean,std

train_dataset = MyDataset(train_list,ImageTransform(resize,mean,std),phase = 'train')
val_dataset = MyDataset(val_list,ImageTransform(resize,mean,std),phase = 'val')
