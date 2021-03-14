from utils2 import *

js = JobShop("instances/ft20")
solution = js.solve(GreedySolver("EST_LRPT"))
print("makespan pour ft20 : ")
print(solution.gant_duration)