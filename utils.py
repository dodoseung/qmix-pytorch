"""
Utility functions for Pytorch
"""
import torch
import datetime;
import os
import yaml

best_loss = float('inf')
directory = ''

def save_model(model_name=None, epoch=-1, model=None, optimizer=None, loss=-1, config=None):
    """Save the model with epoch, criterion, optimizer, and loss.
    Args:
        model_name (string): Model name
        epoch (int): Epoch
        model (torch): Model
        optimizer (torch): Model optimizer
        loss (float): Loss
    """
    # Get the global value
    global best_loss, directory
    
    # Set the model name
    currnet_time = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    
    if epoch == 0:
        # Set the directory path
        DIR_CURRENT = os.path.dirname(os.path.realpath(__file__))
        DIR_PATH = f'{DIR_CURRENT}/models/{model_name}/{currnet_time}'
        directory = DIR_PATH

        # Make the directory
        if not os.path.exists(DIR_PATH):
            os.makedirs(DIR_PATH)
            
        # Save the configuration
        if config:
            save_yaml(f'{DIR_PATH}/config.yml', config)
    else:
        DIR_PATH = directory
    
    # Set the model path
    PATH = f'{DIR_PATH}/{model_name}_epoch_{epoch}_loss_{loss:.3f}_time_{currnet_time}.pt'
    BEST_PATH = f'{DIR_PATH}/{model_name}_above_best.pt'
    LAST_PATH = f'{DIR_PATH}/{model_name}_above_last.pt'
    
    # Save the model
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, PATH)
    # Log
    print(f'Save the model at {DIR_PATH}\t[epoch] {epoch}\t[loss] {loss:.3f}')
    
    # Save the last model
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, LAST_PATH)
    
    # Save the best model
    if loss < best_loss: 
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, BEST_PATH)
        # Log
        print(f'Save the best model at {DIR_PATH}\t[epoch] {epoch}\t[loss] {loss:.3f} \t[Past best] {best_loss:.3f}')
        
        # Set the best loss
        best_loss = loss

        
def load_model(model_path, model=None, optimizer=None):
    """Load the model with epoch, criterion, optimizer, and loss.
    Args:
        model_path (string): Path for model
        epoch (int): Epoch
        model (torch): Model
        optimizer (torch): Model optimizer
        loss (float): Loss
    """
    # Set the model path
    PATH = model_path
    
    # Get the checkpoint
    checkpoint = torch.load(PATH)
    
    # Load the model
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded the model from {PATH} successfully")
    
    # Load the optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded the optimizer from {PATH} successfully")
        
    # Load the epoch and loss
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

def save_yaml(yaml_path, config):
    """Save the yaml configuration file.
    Args:
        yaml_path (string): Path for the yaml file
        config (yaml): yaml data
    """
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print("Save the yaml file in {}".format(yaml_path))
    
def load_yaml(yaml_path):
    """Load the yaml configuration file.
    Args:
        yaml_path (string): Path for the yaml file
    """
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        
    print("Load the yaml file from {}".format(yaml_path))
    return config