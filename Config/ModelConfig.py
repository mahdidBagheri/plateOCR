import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
epochs = 50
learning_rate = 0.01
lr_decay_rate = 0.90