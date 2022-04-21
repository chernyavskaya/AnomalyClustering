import os,sys
import os.path as osp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skl
plt_style = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/utils/adfigstyle.mplstyle'
plt.style.use(plt_style)


def loss_curves(train_valid_loss_dict, save_path):
    if train_valid_loss_dict=={}:
        print('Empty dictionary with losses, probably because there was only 1 epoch. Nothing is plotted.')
        return 0
    keys = train_valid_loss_dict.keys()
    loss_names = [k.replace('Training ','').replace('Validation ','') for k in keys]
    loss_names = set(loss_names)
    train_loss_dict, valid_loss_dict = {},{}
    for key in loss_names:
        train_loss_dict[key] = train_valid_loss_dict['Training '+key]
        valid_loss_dict[key] = train_valid_loss_dict['Validation '+key]
    for key in train_loss_dict.keys():
        epochs = train_loss_dict[key][0]
        train_loss = train_loss_dict[key][1]
        valid_loss = valid_loss_dict[key][1]
        plt.plot(epochs, train_loss, valid_loss)
        plt.xticks(epochs)
        ax = plt.gca()
        #ax.set_yscale('log')
        if max(epochs) < 60:
           ax.locator_params(nbins=10, axis='x')
        else:
            ax.set_xticks(np.arange(0, max(epochs), 20))
        plt.xlabel("Epochs")
        plot_name =key
        plt.ylabel(plot_name)
        plt.legend(['Train', 'Validation'])
        plt.subplots_adjust(left=0.15)
        plt.savefig(osp.join(save_path, 'loss_curves_{}.pdf'.format(plot_name.replace(' ','_'))))
        plt.savefig(osp.join(save_path, 'loss_curves_{}.png'.format(plot_name.replace(' ','_'))))
        plt.close()



