import numpy as np

from utils.confusion_matrix import generate_conf_matrix


def test(model, test_dataloader, test_dataset, device):
    model.eval()
    correct = 0
    predicted = []
    actual = []

    for batch in test_dataloader:
        batch = batch.to(device)
        pred = model(batch).max(1)[1]
        correct += pred.eq(batch.y).sum().item()

        predicted.extend(list(pred.cpu().numpy()))
        actual.extend(list(batch.y.cpu().numpy()))

    print(f'Test accuracy: {correct / len(test_dataset)}')
    generate_conf_matrix(predicted, actual)