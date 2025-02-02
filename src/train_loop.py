import torch
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.data.utils import show_prediction

def train(model, train_dataloader, test_dataloader, device, optimizer, num_epochs, criterion, save_path):
    torch.cuda.empty_cache()
    gc.collect()
    
    history = {'train_loss': [], 'val_loss': []}
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_loader_tqdm = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, masks in train_loader_tqdm:
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            if not torch.is_tensor(outputs):
                outputs = outputs['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() * inputs.size(0)
            epoch_train_loss += batch_loss
            train_loader_tqdm.set_postfix(loss=loss.item())
        
        epoch_train_loss /= len(train_dataloader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        model.eval()
        epoch_val_loss = 0.0
        test_loader_tqdm = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
        with torch.no_grad():
            for inputs, masks in test_loader_tqdm:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                if not torch.is_tensor(outputs):
                    outputs = outputs['out']
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item() * inputs.size(0)
                test_loader_tqdm.set_postfix(loss=loss.item())
        
        epoch_val_loss /= len(test_dataloader.dataset)
        history['val_loss'].append(epoch_val_loss)

        checkpoint_path = f"{save_path}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        show_prediction(test_dataloader, model)
        print(f"Train Loss = {epoch_train_loss}. Test Loss = {epoch_val_loss}")
    
    return history