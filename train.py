##############################################################################################################
# @Author       : SujayKumar Reddy M, R&D IT 
# Date Created  : 8th May, 2024
# File Usage    : Performing the Clustering Operations on Genome data
# Assumption    : Genome is a Sequence data, but for the clustering operation, we use an .csv file.
##############################################################################################################


# Imports
import os
import argparse
from src.clustering.normal_cluster import NormalCluster
import warnings

parser = argparse.ArgumentParser(description="Clustering Pipeline for different algorithms")
parser.add_argument("algorithm",metavar="algorithm",type=str,default="kmeans",help="Specify the Algorithm for Clustering")
parser.add_argument("--include__hyperparams",metavar="hyperparam",type=bool,default=True,help="Hyperparameter Tuning for all/specific Algorithm")
parser.add_argument("--add_viz",metavar="add_viz",type=bool,default=False,help="This parameter adds the visualization, but uses PCA for data dimensionality Reduction")

# Still Checking on Data Sampling parameter for Unsupervised Learning
# parser.add_argument()

args = parser.parse_args()

with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nc = NormalCluster(args)
        print(nc.data)










