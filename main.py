from utils.JobShop import *
from utils.Graphe import *
from datetime import timedelta

js = JobShop(filename="instances/ft06")
sol2 = js.TabooSolver(30, timedelta(seconds=60),5)


print(sol2.duration)