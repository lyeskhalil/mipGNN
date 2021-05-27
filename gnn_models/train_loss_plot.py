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
    # "L_n200_p0.02_c500_train",
]

sns.set_context(context="paper", font_scale=1.5, rc=None)
sns.set(font="Arial")

###############
# Set architecture here.
architecture = "EC"  # ["ECS", "GINS", "SGS", "EC", "GIN", "SG"]
######################

bias = [0.0, 0.001, 0.1]

cols = ['Bias {}'.format(col) for col in bias]
rows = ['{}'.format(row) for row in name_list]
fig, axes = plt.subplots(len(name_list), len(bias), figsize=(15, 50), sharey=False)
fig.suptitle("Architecture: " + architecture)
plt.setp(axes.flat, xlabel='Epochs', ylabel='NLL')

pad = 5  # in points

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

for ax, row in zip(axes[:, 0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation="90")

# fig.set(xlabel='Epoch', ylabel='NLL')
for i, name in enumerate(name_list):
    for j, b in enumerate(bias):
        log = architecture + "_" + name + str(b) + ".log"
        print("./model_new/logs/" + log)

        data = np.loadtxt(fname="./model_new/logs/" + log, delimiter=",")
        data = pd.DataFrame(data,
                            columns=["epoch", "train_loss", "train_acc", "train_f1", "train_pr", "train_re", "val_acc",
                                     "val_f1", "val_pr", "val_re", "best_val", "test_acc", "test_f1", "test_pr",
                                     "test_re"])

        sns.lineplot(ax=axes[i, j], x='epoch', y='train_loss', data=data, linewidth=3.5)

fig.tight_layout()

fig.subplots_adjust(left=0.15, top=0.95)
plt.show()
