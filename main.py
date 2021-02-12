from utils2 import *


g1 = Graphe()
g1.add(["A", "B", "C", "D", "E", "F", "G"])
g1.link(["A", "B", "C", "D"])
g1.link(["B", "E", "F"])
g1.link(["G", "A"])
g1.link(["E", "C"])
g1.link(["F", "D"])


print(g1.topological_list())
print(g1.topological_list_2())

exit()

js = JobShop("instances/ft06")
solution = js.solve(GreedySolver("EST_LRPT"))
print(solution.gant_duration)