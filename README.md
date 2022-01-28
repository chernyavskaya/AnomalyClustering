# AnomalyClustering

 - Clustering using Deep Embedded Clustering [arXiv:1511.06335, https://www.ijcai.org/proceedings/2017/243] is set up in 'IDEC'
 - The framework is split into several modules for data preprocessing, training utils, etc.
 - Autoencoder for iDEC is setup with graph neural network architecture using pytorch-geometric, and with feed-forward neural network. The iDEC class is common to both types of architecture, although the training scripts are separate. 
 - Standalone examples can be found in standalone scripts. Only the MNIST version is supposed to be used as standalone (used as a toy only).
 The feed-forward and graph neural networks setup for our particle dataset are also present there, however will not be any longer updated due to redundancy with the main framework. 
- Several notebooks for testing different set-ups and showing how to use the framework are in 'notebooks'
