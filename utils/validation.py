import numpy as np
import torch

from utils.io.save_wrong_result import save_wrong_result


# def validate(model, data):
#     total = 0
#     correct = 0
#
#     for batch in data:
#         batch.to(device)
#         x = model(batch)
#         value, pred = torch.max(x, 1)
#         pred = pred.data.cpu()
#         total += float(x.size(0))
#         correct += float(torch.sum(pred == labels))
#
#     return correct * 100. / total


def validate_cel(model, data, cel, device):
    total = 0
    correct = 0
    results = []

    with(torch.set_grad_enabled(False)):
        for batch in data:
            batch.to(device)
            x = model(batch)
            results.append(cel(x, batch.y))

            value, pred = torch.max(x, 1)
            total += float(x.size(0))
            correct += pred.eq(batch.y).sum().item()

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
