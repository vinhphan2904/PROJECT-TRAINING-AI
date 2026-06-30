from lib import *
from imagetransform import ImageTransform
from normalize import *
from cnn import MyCNN
from mapping import label_to_idx,idx_to_label
from datalist import val_list

def has_image():
    return len(os.listdir('yolo/data')) > 0

device = torch.device(
    'mps' if torch.backends.mps.is_available() else 'cpu'
)

checkpoint = torch.load('best_checkpoint.pth',map_location = device)
model = MyCNN().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
transform_img = ImageTransform(resize,mean,std)
path = 'yolo/data'
new_path = 'real_data'
json_path = 'core/data.json'
while True:
    if has_image():
        file = sorted(os.listdir(path))
        file_name = file[0]
        time_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_path = os.path.join(path,file_name)
        img = Image.open(img_path).convert('RGB')
        img = transform_img(img,phase = 'val').to(device)
        data = img.unsqueeze(0)
        state_health = ''
        confidence = 0.0
        with torch.no_grad():
            outputs = model(data)
            _,pred = torch.max(outputs,dim = 1)
            prob = torch.softmax(outputs,dim = 1)
            state_health = idx_to_label[pred.item()]
            confidence = prob[0][pred.item()].item()
        print(state_health)
        print(confidence)
        print(time_date)
        print("YES")
        if state_health != 'Healthy' and confidence > 0.8:
            shutil.copy(img_path,new_path)
            if os.path.exists(json_path):
                with open(json_path,'r') as f:
                    try:
                        old_data = json.load(f)
                    except Exception as e:
                        old_data = []
                        print(e)
            else: 
                old_data = []
            data = {
                    'Time' : time_date,
                    'State_health': state_health,
                    'Confidence': confidence,
                    'Location_img' : img_path
                }
            old_data.append(data)
            with open(json_path,'w') as f:
                json.dump(old_data,f,indent = 4)
        os.remove(os.path.join(path,file_name))
    else:
        print("Wait loading img")
    time.sleep(2)
print("END")