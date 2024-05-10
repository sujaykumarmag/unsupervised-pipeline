


from src.model.model import get_models
import pandas as pd



class NormalCluster():

    def __init__(self,args):
        self.alg = args.algorithm
        self.hyp = args.include__hyperparams
        self.vis = args.add_viz
        self.data = pd.read_csv("datasets/matrix.csv")
        self.data = self.data.drop(["Population code","Unnamed: 0"],axis=1)

        if self.hyp is True:
            self.get_select_alg(self.alg)



    def select_alg(self):
        """
        1. Needed to find the functionality added is available with us {algs_added}
        2. Ability to provide an Exception rather print() exit() in else block // Still Needed
        """
        if self.alg == "all" or self.alg == "All":
            self.get_all()
        else:
            self.get_select_alg(self.alg)
            


    def get_all():
        models = get_models().values()
        for model in models:
            model.train()

    
    def get_select_alg(self,alg):
        models = get_models()
        if alg in list(models.keys()):
            print(models[alg])
            models[alg].train(self.data,self.vis)
        else:
            print(f"The {self.alg} provided by the user not provided in the functionality \n {models.keys()}")
            exit()


        


        

        