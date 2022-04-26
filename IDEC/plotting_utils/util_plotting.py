import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skl
#plt_style = '/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/utils/adfigstyle.mplstyle'
#plt.style.use(plt_style)


def plot_hist( data, xlabel, ylabel, title, plotname=None, legend=[], ylogscale=True ):
    fig = plt.figure( )
    plot_hist_on_axis( plt.gca(), data, xlabel, ylabel, title, legend, ylogscale )
    if legend:
        plt.legend()
    plt.tight_layout()
    if plotname:
        fig.savefig(plotname + '.png')
        fig.savefig(plotname + '.pdf')
        plt.close()


def plot_features( datas, xlabel, ylabel, title, plotname='', legend=[], ylogscale=True ):
    if isinstance(datas,list) :
        num_feats = datas[0].shape[-1]
    else : 
        num_feats = datas[0].shape[-1] if datas.ndim==3 else datas.shape[-1]
    for i in range(num_feats):
        if type(xlabel)==str : 
            xlabel_plot = xlabel+' {}'.format(i)
        else : 
            xlabel_plot = xlabel[i]
        if isinstance(datas,list) :
            plot_data = [data[:,i] for data in datas] 
        else :
            plot_data = [data[:,i] for data in datas] if datas.ndim==3 else [datas[:,i]]
        plot_hist_many(plot_data, xlabel_plot, ylabel, title, plotname=plotname+xlabel_plot+'_log', legend=legend, ylogscale=ylogscale )
        #plot_hist_many(plot_data, xlabel_plot, ylabel, title, plotname=plotname+xlabel_plot, legend=legend, ylogscale=False )

        
def plot_2dhist( data_x, data_y,xlabel, ylabel, title, plotname=None,cmap=plt.cm.Reds):
    fig = plt.figure( )
    max_score_x =np.quantile(data_x,0.95)
    min_score_x = np.quantile(data_x,0.05)
    max_score_y =np.quantile(data_y,0.95)
    min_score_y = np.quantile(data_y,0.05)
    bins_x = np.linspace(2*min_score_x if min_score_x<0 else 0.5*min_score_x,2*max_score_x if max_score_x>0 else 0.5*max_score_x,100)
    bins_y = np.linspace(2*min_score_y if min_score_y<0 else 0.5*min_score_y,2*max_score_y if max_score_y>0 else 0.5*max_score_y,100)
    kwargs={'cmap':cmap,'density':True}
    im = plt.hist2d( data_x,data_y, bins=(bins_x,bins_y), **kwargs)
    fig.colorbar(im[3])
    plt.ylabel( ylabel )
    plt.xlabel( xlabel )
    plt.title( title, fontsize=10 )
    plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.tight_layout()
    if plotname:
        fig.savefig(plotname + '.png')
        fig.savefig(plotname + '.pdf')
        plt.close()




def plot_hist_many( datas, xlabel, ylabel, title, plotname=None, legend=[], ylogscale=True ):
    fig = plt.figure( )
    #max_score = np.max([1.1*np.quantile(x,0.97) for x in datas])
    #min_score = np.min([0.9*np.quantile(x,0.03) for x in datas])
    max_score = np.max([np.max(x) for x in datas])
    min_score = np.min([np.min(x) for x in datas])
    kwargs={'linewidth':2.3, 'fill':False, 'density':True,'histtype':'step'}
    bins = 100
    for i,data in enumerate(datas):
        if ylogscale:
            plt.semilogy( nonpositive='clip')
        if i==0:
            _,bins,_ = plt.hist( data, bins=bins, range=(min_score,max_score),label=legend[i], **kwargs)
        else :
            plt.hist( data, bins=bins, linestyle='--', label=legend[i],**kwargs)
    plt.ylabel( ylabel )
    plt.xlabel( xlabel )
    plt.title( title, fontsize=10 )
    plt.tick_params(axis='both', which='minor', labelsize=8)
    if legend:
        plt.legend(bbox_to_anchor=(1., 1.),fontsize=15)
    plt.tight_layout()
    plt.show()
    if plotname:
        fig.savefig(plotname + '.png')
        fig.savefig(plotname + '.pdf')
        plt.close()

def plot_scatter_many( datas, xlabel, ylabel, title, plotname=None, legend=[] ):
    fig = plt.figure( )
    for i,data in enumerate(datas):
        plt.scatter( data[:,0],data[:,1],label=legend[i], alpha=0.3)
    plt.ylabel( ylabel )
    plt.xlabel( xlabel )
    plt.title( title, fontsize=10 )
    plt.tick_params(axis='both', which='minor', labelsize=8)
    if legend:
        plt.legend(bbox_to_anchor=(1., 1.),fontsize=15)
    plt.tight_layout()
    if plotname:
        fig.savefig(plotname + '.png')
        fig.savefig(plotname + '.pdf')
        plt.close()



def plot_hist_on_axis( ax, data, xlabel, ylabel='count', title='', legend=[], ylogscale=True ):
    bin_num = 70
    alpha = 0.85
    if ylogscale:
        ax.set_yscale('log', nonposy='clip')
    ax.hist( data, bins=bin_num, normed=True, alpha=alpha, histtype='stepfilled', label=legend )
    ax.set_ylabel( ylabel )
    ax.set_xlabel( xlabel )
    ax.set_title( title, fontsize=10 )
    ax.tick_params(axis='both', which='minor', labelsize=8)
    #ax.set_ylim(bottom=1e-7)


def plot_hist_2d( x, y, xlabel, ylabel, title, plotname=None):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    im = plot_hist_2d_on_axis( ax, x, y, xlabel, ylabel, title )
    fig.colorbar(im[3])
    plt.tight_layout()
    if plotname:
        fig.savefig(plotname + '.png')
        fig.savefig(plotname + '.pdf')
        plt.close()


def plot_hist_2d_on_axis( ax, x, y, xlabel, ylabel, title ):
    im = ax.hist2d(x, y, bins=100, norm=colors.LogNorm())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    #ax.set_ylim(top=70.)
    return im


def plot_graph( data, xlabel, ylabel, title, plotname=None, legend=[], ylogscale=True):
    fig = plt.figure()
    if ylogscale:
        plt.semilogy( data )
    else:
        plt.plot( data )
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    if legend: plt.legend(legend, loc='upper right')
    plt.title( title )
    plt.tight_layout( )
    if plotname:
        fig.savefig(plotname + '.png')
        fig.savefig(plotname + '.pdf')
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

    nice_colors = ['#7a5195','#ef5675','#3690c0','#ffa600','#67a9cf','#014636', '#016c59']
    for y_true, loss, label,color in zip(class_labels, losses, legend,nice_colors):
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



def plot_phi_correlations(input_feats,pred_feats,title='',plotname=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6),sharey=True)
    binning = np.linspace(-1,1,100)
    mappables=[]
    for data in [input_feats,pred_feats]:
        h,xedges,yedges = np.histogram2d(data[:,0], data[:,1], bins=[binning,binning],density=True)
        mappables.append(h)
    vmin = np.min(mappables)
    vmax = np.max(mappables)
    for ax,g in zip(axs.ravel(),mappables):
        im = ax.imshow(g,vmin=vmin, vmax=vmax, cmap=plt.cm.BuPu)
        ax.set_title('Input {}'.format(title))
        ax.set_xlabel('cos(phi)')
        ax.set_ylabel('sin(phi)')
    axs[0].set_title('Input {}'.format(title))
    axs[1].set_title('Predicted {}'.format(title))
    fig.colorbar(im,fraction=0.02*h.shape[1]/h.shape[0] ,ax=axs.ravel())
    plt.show()
    if plotname:
        fig.savefig(plotname + '.png')
        fig.savefig(plotname + '.pdf')
        plt.close()