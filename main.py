from utils.JobShop import *
from utils.Graphe import *
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



js = JobShop(filename="instances/ft06")

sol_taboo =dict()
M = [10,20,30,40,50,60,70]
D = [1,5,10,15,20,25,30]

results = np.zeros((len(M),len(D)))

for i,maxiter in enumerate(M):
    print('maxiter',maxiter)
    for j,d in enumerate(D):
        print('duree',d)
        results[i,j] = js.TabooSolver(maxiter, timedelta(seconds=120),d).duration
        print(results[i,j])
        

sns.heatmap(results, vmin=np.min(results), vmax=np.max(results))

        