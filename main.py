from utils.JobShop import *

js = JobShop()

sol = js.heuristique_gloutonne("EST_SPT")
print(sol.get_invertibles())
print("solution originale : ")
print(sol)
print("voisins : ")
for s in sol.solution_neighbors():
    print(s)
    print("makespan : ", s.duration)
