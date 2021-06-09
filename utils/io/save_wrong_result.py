import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

DIR_NAME = "wrong"
# replace previous directory
if os.path.exists(DIR_NAME):
    shutil.rmtree(DIR_NAME)

os.mkdir(DIR_NAME)


def save_wrong_result(id, image, predicted, actual):
    i = image.cpu()
    i = np.squeeze(i)
    # reshape for squeezenet
    i = i[0].reshape(-1, 224)
    i = i * 255
    path = f"{DIR_NAME}/{id}_pred_{predicted}_act_{actual}.png"
    plt.imsave(path, i, cmap="gray")
