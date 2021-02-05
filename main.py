from utils.JobShop import *

js = JobShop()

sol = js.heuristique_gloutonne("EST_SPT")
print(sol.duration)
print(sol.new_neighbors())
