import os
import torch
from torch.nn import MSELoss
import numpy as np
import torch.optim as optim
from time import sleep
from tqdm import tqdm
from torch import save


def step(data_set, model, optimizer, criterion, epoch, mode='train'):
    rmses = []
    data_size = len(data_set)
    progress_bar = tqdm(np.arange(data_size), desc=f'{mode} epoch {epoch}', total=data_size)
    
    if mode == 'train':
        model.train()
    else:
        model.eval()

    for i, (x, y) in enumerate(data_set):
        out = model(x)
        loss = criterion(out, y)
        rmses.append(np.sqrt(loss.item()))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del x, y, loss
        else:
            del x, y, loss
        if i < len(data_set) - 1:
            progress_bar.update()

    rmse = np.mean(rmses)
    
    progress_bar.set_postfix({'rmse': rmse})
    progress_bar.update()

    return rmse


def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target)) 


def train_model(model, model_save_dir, train_dataset, val_dataset=None, num_epochs=40, lr=1e-4, 
                early_stopping_threshold=10, l2=1e-4):
    model_parameters_path = os.path.join(model_save_dir, 'model_parameters')
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    criterion = MSELoss()
    # criterion = MAPELoss()   
    # criterion = CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam([{'params': model.parameters()}], lr=lr, weight_decay=l2)

    min_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        for mode, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
            if dataset is not None:
                loss = step(dataset, model, optimizer, criterion, epoch, mode)
        
        # Prevent printing of progress bars from messing up
        print(' ')
        sleep(0.1)

        # Save model state if it has improved
        if loss < min_loss:
            save(model.state_dict(), model_parameters_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        min_loss = min(min_loss, loss)

        # Stop training early if the model has not improved for 'early_stopping_threshold' epochs.
        if epochs_without_improvement >= early_stopping_threshold:
            print(f'Early stopping after {epoch} epochs')
            break
