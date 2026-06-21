from lib import *
from datasetlist import train_dataset,val_dataset
from config import *

train_dataloader = data.DataLoader(train_dataset,BATCH_SIZE,shuffle=True)
val_dataloader = data.DataLoader(val_dataset,BATCH_SIZE,shuffle=False)

dataloader_dict = {
    'train' : train_dataloader,
    'val' : val_dataloader
}
