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



def prepare_neg_pos_losses(loss_bg,loss_sig,true_labels_sig):
    signal_classes = np.unique(true_labels_sig[:,0])
    neg_class_losses = [loss_bg]*len(signal_classes)
    pos_class_losses = []
    for sig in signal_classes:
        pos_class_losses.append(loss_sig[true_labels_sig[:,0]==sig])
    return neg_class_losses,pos_class_losses
    

def get_label_and_score_arrays(neg_class_losses, pos_class_losses):
    labels = []
    losses = []

    for neg_loss, pos_loss in zip(neg_class_losses, pos_class_losses):
        labels.append(np.concatenate([np.zeros(len(neg_loss)), np.ones(len(pos_loss))]))
        losses.append(np.concatenate([neg_loss, pos_loss]))

    return [labels, losses]


def plot_roc(neg_class_losses, pos_class_losses, legend=[], title='ROC', legend_loc='best', plot_name='ROC', fig_dir=None, xlim=None,ylim=None, log_x=True,vline_threshold=1e-5):

    class_labels, losses = get_label_and_score_arrays(neg_class_losses, pos_class_losses) # neg_class_loss array same for all pos_class_losses

    aucs = []
    fig = plt.figure(figsize=(7, 7))
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75') #line that corresponds to a random decision

    colors = ['#7a5195','#ef5675','#3690c0','#ffa600','#67a9cf','#014636', '#016c59']
    for y_true, loss, label,color in zip(class_labels, losses, legend,colors):
        fpr, tpr, threshold = skl.roc_curve(y_true, loss)
        aucs.append(skl.roc_auc_score(y_true, loss))
        if log_x:
            plt.loglog(fpr,tpr, c=color, label=label + " (AUC = " + "{0:.2f}".format(aucs[-1]) + ")")
        else:
            plt.semilogy( fpr, tpr,c=color,label=label + " (AUC = " + "{0:.2f}".format(aucs[-1]) + ")")

    #plt.grid()
    plt.vlines(vline_threshold, 0, 1, linestyles='--', color='lightcoral')
    if xlim:
        plt.xlim(left=xlim,right=1.)
    if ylim:
        plt.ylim(bottom=ylim,top=1.)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=legend_loc,fontsize=14)
    plt.tight_layout()
    plt.title(title)
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + '.png'), bbox_inches='tight' )
        plt.close(fig)
    plt.show()

    return aucs