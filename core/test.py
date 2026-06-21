from lib import *
from imagetransform import ImageTransform
from normalize import *
from cnn import MyCNN
from mapping import label_to_idx,idx_to_label
from datalist import val_list

device = torch.device(
    'mps' if torch.backends.mps.is_available() else 'cpu'
)
path = random.choice(val_list)
print(path)
img = Image.open(path).convert('RGB')
img.show()
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