from sklearn.decomposition import PCA

class ClusteringTemplate():

    def __init__(self):
        pass

    def train(self,data):
        pass

    def train_with_hyp(self,data):
        pass

    def vis_plots(self):
        pass

    def dimension_red(self,data):
        pca = PCA(n_components=2)
        data = pca.fit_transform(data) 
        return data