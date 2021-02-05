from utils.JobShop import *
from datetime import timedelta
js = JobShop()

#sol = js.heuristique_gloutonne("EST_SPT")
sol = js.TabooSolver(maxIter = 20,dureeTaboo=timedelta(seconds = 10))
print(sol.duration)
print(sol.new_neighbors())
