from lib import *
from mapping import idx_to_label,label_to_idx
class MyDataset(data.Dataset):
    def __init__(self,data_list,transform = None,phase = 'train'):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = Image.open(img_path)
        img = self.transform(img,self.phase)
        path = os.path.basename(os.path.dirname(img_path))
        label = label_to_idx[path]
        return img, label