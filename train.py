from Dataset.Dataset import PlateDataset
from torch.utils.data import DataLoader
from Model.Model import OCRNet
import torch
from Model.Loss import CTCLoss
from Model.Learner import Learner
from Config.ModelConfig import epochs, device, learning_rate
from Config.DatasetConfig import test_dataset_size, train_dataset_size

if __name__ == "__main__":
    train_dataset = PlateDataset(train_dataset_size)
    test_dataset = PlateDataset(test_dataset_size)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    model = OCRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = CTCLoss().to(device)

    learner = Learner(model, loss, optimizer, train_loader, test_loader)
    for epoch in range(epochs):
        train_results = learner.run_epoch(epoch, val=False)
        test_results = learner.run_epoch(epoch, val=True)
