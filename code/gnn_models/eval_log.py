import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, '.')

name_list = [
    "p_hat300-2.clq_train",
    "gisp_C250.9.clq_train",
    "keller4.clq_train",
    "hamming8-4.clq_train",
    "gen200_p0.9_55.clq_train",
    "gen200_p0.9_44.clq_train",
    "C125.9.clq_train",
    "p_hat300-1.clq_train",
    "brock200_4.clq_train",
    "brock200_2.clq_train",
    "L_n200_p0.02_c500_train",
]

name = name_list[2]
bias = 0.0  # [0.0, 0.001, 0.1]
architecture = "SG"  # ["ECS", "GINS", "SGS", "EC", "GIN", "SG"]

log = architecture + "_" + name + str(bias) + ".log"

print("./model_new/logs/" + log)
# print(os.path.exists("../gnn_models/model_new/logs/" + log))

# [epoch, train_loss, train_acc, train_f1, train_pr, train_re, val_acc, val_f1, val_pr, val_re, best_val, test_acc, test_f1, test_pr, test_re])
data = np.loadtxt(fname="./model_new/logs/" + log, delimiter=",")
data = pd.DataFrame(data, columns=["epoch", "train_loss", "train_acc", "train_f1", "train_pr", "train_re", "val_acc",
                                   "val_f1", "val_pr", "val_re", "best_val", "test_acc", "test_f1", "test_pr",
                                   "test_re"])

data = data[["epoch", "train_loss", "train_loss"]]

data_new = pd.melt(data, 'epoch', var_name='Score',
                   value_name='Value')

fig = sns.lineplot('epoch', 'Value', hue='Score',
                   data=data_new)

# fig.set_axis_labels('Epochs', 'Score')
plt.tight_layout()

plt.show()
