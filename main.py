from utils.JobShop import *

js = JobShop()
print(js.heuristique_gloutonne_2("SPT"))
print(js.heuristique_gloutonne_2("LPT"))
print(js.heuristique_gloutonne_2("SRPT"))
print(js.heuristique_gloutonne_2("LRPT"))

exit()
sol = Solution.from_ressource_matrix(js, js.heuristique_gloutonne())
print(sol)
print(sol.is_realisable())
print(sol.get_duration())
