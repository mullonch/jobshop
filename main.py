from utils2 import *

js = JobShop("instances/ft06")
solution = js.solve(GreedySolver("EST_LRPT"))
print("makespan pour ft06 : ")
print(solution.gant_duration)