# CNNs for Structural MRI Data
This repository contains a flexible set of scripts to run convolutional neural networks (CNNs) on structural brain images.  
It was written using python 3.6.3 and tensorflow 1.4.0. It requires tensorflow (and all dependencies).  
It is run using: `python -m run_scripts.runCustomCNN` from the code directory. The script also takes the following options:
```
required arguments:
  --gpuMemory GPUMEMORY
                        A float between 0 and 1. The fraction of available
                        memory to use.
  --numSteps NUMSTEPS   The number of steps to train for.
  --scale SCALE         The scale at which to slice dimensions. For example, a
                        scale of 2 means that each dimension will be devided
                        into 2 distinct regions, for a total of 8 contiguous
                        chunks.
  --type TYPE           One of: traditional, reverse
  --summaryName SUMMARYNAME
                        The file name to put the results of this run into.
  --data DATA           One of: PNC, PNC_GENDER, ABIDE1, ABIDE2, ABIDE2_AGE
optional arguments: 
  --poolType POOLTYPE
						The type of the pooling layer used inside the network.
						One of: MAX, AVG, STRIDED, NONE
  --sliceIndex SLICEINDEX
                        Set this to an integer to select a single brain region
                        as opposed to concatenating all regions along the
                        depth channel.
  --align ALIGN         Set to 1 to align channels.
  --numberTrials NUMBERTRIALS
                        Number of repeated models to run.
  --padding PADDING     Set this to an integer to crop the image to the brain
                        and then apply `padding` amount of padding.
  --batchSize BATCHSIZE
                        Batch size to train with. Default is 4.
  --pheno PHENO         Specify 1 to add phenotypics to the model.
  --validationDir VALIDATIONDIR
                        Checkpoint directory to restore the model from.
						If not specified, program will check the default
						directory for stored parameters.
  --regStrength REGSTRENGTH
                        Lambda value for L2 regularization. If not specified,
                        no regularization is applied.
  --learningRate LEARNINGRATE
                        Global optimization learning rate. Default is 0.0001.
  --maxNorm MAXNORM     Specify an integer to constrain kernels with a maximum
                        norm.
  --dropout DROPOUT     The probability of keeping a neuron alive during
                        training. Defaults to 0.6.
  --dataScale DATASCALE
                        The downsampling rate of the data. Either 1, 2 or 3.
                        Defaults to 3.
  --pncDataType PNCDATATYPE
                        One of AVG, MAX, NAIVE. Defaults to AVG. If set,
                        dataScale cannot be specified.
  --listType LISTTYPE   Only valid for ABIDE and ADHD. One of strat or site.
  --depthwise DEPTHWISE
						Set to 1 use depthwise model instead of standard model
  --skipConnection SKIPCONNECTION
						Set to 1 to allow skip connection layer, add residuals
						to the network (like ResNet).
```
  
  The scripts assume that you have the following directories, which you will have to create yourself:   
  `brain_age_prediction/summaries/`   
  `brain_age_prediction/checkpoints/`   
It also requires that the data directories exist in the config files, and that those data directories actually contain data. 

  
  
  The code itself is organized into several directories.
  
### model/
  This directory contains two files related to building the model, both of  
  which can be used independently of the rest of the repository: `buildCommon.py` and `buildCustomCNN.py`.  
  The former contains wrapper functions around tensorflow's 3D model-building functions  
  to create 3D convolutional layers, fully connected layers, batch normalization and pooling layers.  
  The latter contains a single function that aggregates all of the model-building functions  
  into a single, flexible CNN that is largely built by the passed-in parameters.   
   
  
### engine/
  This directory contains most of the heavy-lifting of the code base. `trainCustomCNN.py`  
  takes in options from the command line, and builds the model and loads the datasets based  
  on those parameters. `trainCommon.py` is a file that contains flexible functions related to  
  automatically training a model given a gradient update operation. It trains the model for a  
  specified number of iterations, saving the model that did best on the validation set while training.  
  It also contains functions to output visualizations of arbitrary operations to tensorboard summaries.  
  
### data_scripts/
  `DataSetNPY.py` contains a class that loads in and produces batches of .npy  
  files, which is useful if you have matrix data with arbitrary dimensions. 
  `DataSetBIN.py` contains a class that loads in and produces batches of binary files,  
  which is faster than .npy files, but less flexible. 
  
### run_scripts/
  `runCustomCNN.py` is a simple script that runs the engine scripts.
  
### utils/
  This file contains several helper files.  
  `args.py` takes in command-line arguments.  
  `config.py` reads the config.json file in the code directory.  
  `patches.py` does regional segmentation as described in the paper.  
  `saveModel.py` restores tensorflow models from a given directory.  
  `sliceViewer.py` is a class that allows one to view numpy matrices in dimensions higher than 2D.
  
### placeholders/
  `shared_placeholders.py` contains several functions to return placeholders  
  for fed-in data.
  
### archived/
  This directory contains a wealth of files that were used to run previous experiments.  
  These files are now no longer maintained or have been re-written in an updated script.  
  
