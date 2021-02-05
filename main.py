from utils.JobShop import *
from datetime import timedelta
js = JobShop()


#sol = js.heuristique_gloutonne("EST_SPT")
sol = js.TabooSolver(maxIter = 20,dureeTaboo=timedelta(seconds = 10))
print(sol.duration)
print(sol.new_neighbors())

# sol = js.heuristique_gloutonne("EST_SPT")
# print(sol.get_invertibles())
# print("solution originale : ")
# print(sol)
# print("voisins : ")
# for s in sol.solution_neighbors():
#     print(s)
#     print("makespan : ", s.duration)

