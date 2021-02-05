from utils.JobShop import *

js = JobShop()
sol = js.descente()
n = sol.solution_neighbors()
for s in n:
    print(s)
    print(s.duration)
    print(n[s])
