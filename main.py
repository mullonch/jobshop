from utils.JobShop import *

js = JobShop()
print(js.heuristique_gloutonne_2("SPT"))
print(js.heuristique_gloutonne("SPT"))

exit()
sol = Solution.from_ressource_matrix(js, js.heuristique_gloutonne())
print(sol)
print(sol.is_realisable())
print(sol.get_duration())
