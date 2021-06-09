from model import GCN
from train import train
from utils.get_device import get_device
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MNISTSuperpixels
from config import BATCH_SIZE, TRAINING_DIR, TEST_DIR
import torch_geometric.transforms as T

transform = T.Cartesian(cat=False)

training_dataset = MNISTSuperpixels(TRAINING_DIR, train=True, transform=transform)
test_dataset = MNISTSuperpixels(TEST_DIR, train=False, transform=transform)

data_size = len(training_dataset)
training_dataloader = DataLoader(training_dataset[:int(data_size * 0.7)], batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(training_dataset[int(data_size * 0.7):], batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

device = get_device()
model = GCN(training_dataset).to(device)

train(model, device=device, training_dataloader=training_dataloader,
      validation_dataloader=validation_dataloader)
