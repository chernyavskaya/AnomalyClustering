# AnomalyClustering

 - Clustering using Deep Embedded Clustering [https://arxiv.org/abs/1511.06335, https://www.ijcai.org/proceedings/2017/243] is set up in 'IDEC'
 - The framework is split into several modules for data preprocessing, training utils, etc.
 - Autoencoder for iDEC is setup with graph neural network architecture using pytorch-geometric, and with feed-forward neural network. The iDEC class is common to both types of architecture, although the training scripts are separate. 
 - Standalone examples can be found in standalone scripts. The MNIST version is supposed to be used as standalone and is used as a toy only.
 - Several notebooks for testing different set-ups and showing how to use the framework are in 'notebooks'
