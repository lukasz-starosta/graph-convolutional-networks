def test(model, data, device):
    model.eval()

    data = data.to(device)
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())

    print(f'Test accuracy: {acc}')
