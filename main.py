from utils.JobShop import *
from utils.Graphe import *
from datetime import timedelta

js = JobShop(filename="instances/ft06")
sol1 = js.descente()
sol3 = js.heuristique_gloutonne()
sol2 = js.TabooSolver(30, timedelta(seconds=60),5)

print(sol1.duration)
print(sol2.duration)
print(sol3.duration)