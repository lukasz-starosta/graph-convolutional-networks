from model import GCN
from test import test
from train import train
from utils.get_device import get_device
from torch_geometric.datasets import Planetoid
from config import DATA_DIR

dataset = Planetoid(DATA_DIR, "Cora")
# Get the first and only graph
data = dataset[0]

device = get_device()
model = GCN(dataset).to(device)

model = train(model, device=device, data=data)

test(model, data=data, device=device)

