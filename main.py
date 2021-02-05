from utils.JobShop import *

from datetime import timedelta

js = JobShop(filename="instances/ft06")

sol = js.TabooSolver(maxIter = 20,timeout = timedelta(seconds = 60),dureeTaboo=5)
sol2 = js.heuristique_gloutonne("EST_SPT")

print(sol.duration)
print(sol2.duration)
# sol = js.descente()
# n = sol.solution_neighbors()


# for s in n:
#     print(s)
#     print(s.duration)
#     print(n[s])
