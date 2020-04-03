import torch
import shutil

def save_ckp(state):
    f_path = "/content/checkpoints/ckp_epoch" + str(state['epoch']) + ".pt"
    torch.save(state, f_path)
    print("Checkpoint created\n")

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
