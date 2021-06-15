import time
from config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from torch import nn, optim, cuda
from utils.validation import validate_cel
from utils.io.save_to_file import save_to_file
import matplotlib.pyplot as plt


def train(model, device, data):
    time_start = time.time()

    accuracies = []
    training_losses = []
    validation_losses = []
    max_accuracy = 0

    cel = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        cuda.empty_cache()

        data.to(device)
        pred = model(data)[data.train_mask]
        loss = cel(pred, data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        training_loss = float(loss)
        training_losses.append(training_loss)

        validation_loss, accuracy = validate_cel(model, data, cel, device)
        validation_losses.append(validation_loss.cpu())
        accuracies.append(accuracy)

        if accuracy > max_accuracy:
            best_model = model
            max_accuracy = accuracy

        print(
            f'Epoch: {epoch + 1}, Accuracy: {accuracy}%, Training loss: {training_loss}, Validation loss: {validation_loss}')

    time_end = time.time()
    print(f'Training complete. Time elapsed: {time_end - time_start}s')

    print(f'Saving best model with accuracy: {max_accuracy}')
    save_to_file(best_model, f'model_acc_{max_accuracy}_ep_{EPOCHS}')

    plt.plot(accuracies, label='Accuracy')
    plt.legend()
    plt.show()

    plt.cla()
    plt.plot(training_losses, label='Training losses')
    plt.plot(validation_losses, label='Validation losses')
    plt.legend()
    plt.show()

    return best_model
