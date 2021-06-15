import copy
import os
from torch import save
from config import MODEL_PATH


def save_to_file(model, filename):
    try:
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        save(copy.deepcopy(model).state_dict(), MODEL_PATH + '/' + filename)
        print(f'Saved model to file: {MODEL_PATH}/{filename}')
    except FileNotFoundError:
        print("Couldn't find file")
