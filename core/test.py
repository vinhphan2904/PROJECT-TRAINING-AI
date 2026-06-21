from lib import *
from imagetransform import ImageTransform
from normalize import *
from cnn import MyCNN
from mapping import label_to_idx,idx_to_label
from datalist import val_list

device = torch.device(
    'mps' if torch.backends.mps.is_available() else 'cpu'
)
path = '/Users/phanhuynhtuankhanh/Downloads/Downloads/5b582031-af19-4cc3-bf11-0bdf1fba180f.png'
path1 = val_list[0]
img = Image.open(path1).convert('RGB')
img_transform = ImageTransform(resize,mean,std)
img = img_transform(img,phase = 'val')
img = img.unsqueeze(0).to(device)
model = MyCNN()
checkpoint = torch.load("best_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

with torch.no_grad():
    output = model(img)
    prob = torch.softmax(output,dim = 1)
    _,pred = torch.max(output,dim = 1)

print(idx_to_label[pred.item()])
print(prob)