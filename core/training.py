from lib import * 
from cnn import MyCNN
from dataloader import dataloader_dict
from imagetransform import ImageTransform
from config import * 

def training(model,dataloader_dict,optimizer,loss_ln,device,best_acc,start_epoch = None,num_epochs = None):
    model = model.to(device)
    best_acc = best_acc
    for epoch in range(start_epoch,num_epochs):
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs,labels in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_ln(outputs,labels)
                    _,pred = torch.max(outputs,dim = 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                epoch_loss += loss.item() * inputs.shape[0]
                epoch_corrects += torch.sum(pred == labels)
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects.float() / len(dataloader_dict[phase].dataset)
            print(f'{phase} Loss : {epoch_loss:.4f} Acc : {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc.item()
                    print(f"🔥 New best val acc: {best_acc:.4f} → saving checkpoint")
                    torch.save({
                        'epoch' : epoch,
                        'model_state_dict' : model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'best_acc' : best_acc
                    },'best_checkpoint.pth')


