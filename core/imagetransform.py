from lib import *

class ImageTransform:
    def __init__(self,resize,mean,std):
        self.data_transforms = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(resize,scale = (0.5,1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ]),
            'val' : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        }
    
    def __call__(self,img,phase = 'train'):
        return self.data_transforms[phase](img)
    
    