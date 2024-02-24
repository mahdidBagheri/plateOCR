from Model.Model import OCRNet
from Dataset import DataSynthesis
from torchvision.transforms import v2
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    pass
    word, plate = DataSynthesis.synthesize_license()
    model = OCRNet()

    transforms = v2.Compose([v2.ToTensor()])
    plate_tensor = transforms(plate)
    plate_tensor = plate_tensor[None, :, :, :]
    X = model(plate_tensor)
    a = 0
