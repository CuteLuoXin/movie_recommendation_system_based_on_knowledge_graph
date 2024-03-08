import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt


def draw_loss_pic(train_loss, test_loss):
    plt.plot([i for i in range(len(train_loss))], [i for i in train_loss], 'b-', label="Train Loss")
    plt.plot([i for i in range(len(test_loss))], [i for i in test_loss], 'r-', label="Test Loss")
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()


def fix_seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


