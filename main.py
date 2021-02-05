from utils.JobShop import *
from datetime import timedelta
js = JobShop()


# sol = js.heuristique_gloutonne("EST_SPT")
# print(sol.V)
sol = js.TabooSolver(maxIter = 20,timeout=timedelta(seconds = 10))
print('duration:',sol.duration)
print('\n')
#print(sol.solution_neighbors())

# sol = js.heuristique_gloutonne("EST_SPT")
# print(sol.get_invertibles())
# print("solution originale : ")
# print(sol)
# print("voisins : ")
# for s in sol.solution_neighbors():
#     print(s)
#     print("makespan : ", s.duration)

