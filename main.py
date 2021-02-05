from utils.JobShop import *
from utils.Graphe import *

g = Graphe()
g += ["A", "B", "C", "D","E"]
print(g.shorter_path_length("A", "E"))