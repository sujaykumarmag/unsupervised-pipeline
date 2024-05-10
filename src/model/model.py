##############################################################################################################
# @Author           : SujayKumar Reddy M, R&D IT 
# Date Created      : 8th May, 2024
# File Usage        : Performing Clustering Operation for all the Unsupervised Algorithms
# Algorithms Used   : KMeans Clustering
##############################################################################################################


from src.model.kmeans import KMeansClustering

models = {}

models["kmeans"] =KMeansClustering()







def get_models():

    """
    It spits out all the models for performing Clustering
    """
    return models

