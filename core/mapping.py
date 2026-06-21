from lib import *

dir_data = './data/dataset'
path = os.path.join(dir_data,'**','*.*')

disease = set()
idx_to_label = {}
label_to_idx = {}

for list in glob.glob(path,recursive=True):
    img = os.path.basename(os.path.dirname(list))
    disease.add(img)

disease = sorted(disease)
for idx,label in enumerate(disease):
    idx_to_label[idx] = label

for idx,label in idx_to_label.items():
    label_to_idx[label] = idx 
