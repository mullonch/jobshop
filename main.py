from utils2 import *


js = JobShop("instances/ft06")
solution = js.solve(GreedySolver("EST_LRPT"))
print(solution.gant_duration)