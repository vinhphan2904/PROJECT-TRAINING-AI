from lib import *
from training import training
from config import *
from cnn import MyCNN
from dataloader import dataloader_dict

if __name__ == '__main__':
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    model = MyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    start_epoch = 0
    best_acc = 0.0

    if os.path.exists("best_checkpoint.pth"):
        checkpoint = torch.load("best_checkpoint.pth", map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]

        print("Loaded checkpoint.")
    else:
        print("No checkpoint found. Training from scratch.")

    loss_ln = nn.CrossEntropyLoss()

    num_epoch = int(input()) + start_epoch

    training(model,dataloader_dict,optimizer,loss_ln,device,best_acc,start_epoch,num_epoch)
    checkpoint = torch.load("best_checkpoint.pth",map_location = device)
    best_acc = checkpoint['best_acc']
    print(best_acc) 