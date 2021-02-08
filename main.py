from utils.JobShop import *
import numpy as np
import seaborn as sns
import random

js = JobShop(filename="instances/ft06")
for i in range(20):
    sol = js.multiple_descente(nb_starts=1)[0]
    print(i, " : ", sol.duration)
    print(sol.problem.jobs)
    print(sol)
    print(sol.gant)
    exit()


exit()
js = JobShop(filename="instances/ft06")
solutions = dict()
for strategy in ["random", "SPT", "LPT", "SRPT", "LRPT","EST_SPT", "EST_LPT"]:
    solutions[strategy] = js.heuristique_gloutonne(strategy)
for key in solutions :
    print(key, " : ", solutions[key].duration)
best = min(solutions.keys(), key=(lambda k: solutions[k].duration))
print("best greedy algorithme : ", best, "(makespan = ", solutions[best].duration, ")")
print("descente à partir de la meilleure solution gloutonne : ")
descente = js.descente(solutions[best])
print("Makespan : ", descente.duration)
print("descentes multiples (20 starters): ")
md = js.multiple_descente(nb_starts=20)
print([s.duration for s in md])
for s in md:
    if s.duration<55:
        print("WTF ?")
        print(s.duration)
        print(s)
        exit()
print("Algos tabous (500 itérations, durée de taboo = 5) : ")
tb = js.tabooSolver(500, float("inf"), 5)
print(tb.duration)

