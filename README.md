# Implementing ANNs with TensorFlow Final Project
## Group 10

This repository contains all code and supplementary materials for our final TensorFlow project. Its contents are as follows:

The **Datasets** directory contains the grapes and bananas datasets created for and used for our project (the creation of which is detailed in the accompanying paper). This directory also contains the prepared grapes object detection dataset which divides the master grapes dataset into train, test, and val partitions for ease of use in the accompanying Colab notebooks. 

The **EN** directory contains our custom EfficientNet model, the scripts for constructing and training it, and the plots of its training performance. 

The **Scripts** directory contains the Python scripts used in the final project, which in the end was two - the script which split the grapes dataset into the appropriate partitions, and the script which created the bananas dataset as a subset of the COCO 2017 dataset. 

The **old** directory contains all files and scripts which were part of work on the project which was ultimately discarded or not needed in the final pipeline, such as the files needed for integrating EfficientNet with the SSD head within the TensorFlow Object Detection API, which we did not in the end succeed in doing. 
