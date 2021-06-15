import numpy as np
import torch


def validate_cel(model, data, cel, device):
    total = 0
    correct = 0
    results = []

    with(torch.set_grad_enabled(False)):
        data.to(device)
        x = model(data)[data.val_mask]
        results.append(cel(x, data.y[data.val_mask]))

        value, pred = torch.max(x, 1)
        total += float(x.size(0))
        correct += pred.eq(data.y[data.val_mask]).sum().item()

    return sum(results) / len(results), correct * 100. / total


def get_predicted_actual(model, data):
    predicted = []
    actual = []

    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()

        predicted.extend(list(pred.numpy()))
        actual.extend(list(labels.numpy()))

    return np.array(predicted), np.array(actual)
