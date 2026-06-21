from lib import *
from core import device
from imagetransform import ImageTransform
from normalize import *
from cnn import MyCNN
from mapping import label_to_idx
with torch.no_grad():
    path = '/Users/phanhuynhtuankhanh/Downloads/Downloads/5b582031-af19-4cc3-bf11-0bdf1fba180f.png'
    img = Image.open(path).convert('RGB')
    img_transform = ImageTransform(resize,mean,std)
    img = img_transform(img,phase = 'val')
    img = img.unsquezze(0).to(device)
    model = MyCNN()
    output = model(img)
    prob = torch.softmax(output,dim = 1)
    _,pred = torch.max(output,dim = 1)
    print(label_to_idx[pred.item()])
    img.show()
    print(prob)