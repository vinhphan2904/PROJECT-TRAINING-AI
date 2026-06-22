from lib import *
from imagetransform import ImageTransform
from normalize import *
from cnn import MyCNN
from mapping import label_to_idx,idx_to_label
from datalist import val_list

def has_image():
    return len(os.listdir('predict/data')) > 0

device = torch.device(
    'mps' if torch.backends.mps.is_available() else 'cpu'
)

while has_image():
    path = 'predict/data'
    file = sorted(os.listdir(path))
    file_name = file[0]
    time = datetime.now().strftime("%Y-%m-%m %H:%M:%S")
    img = os.path.join(path,file_name)
    
