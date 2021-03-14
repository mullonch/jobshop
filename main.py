from utils2 import *

js = JobShop("instances/ft10")
solution = js.solve(GreedySolver("SPT"))
print("makespan pour ft10 : ")
print(solution.gant_duration)
