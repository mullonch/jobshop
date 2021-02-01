from utils.JobShop import *

js = JobShop()
#sol = Solution.from_ressource_matrix(js, [[Task(0,3,0,0), Task(0,2,1,1)],[Task(1,3,1,0), Task(1,2,0,1)], [Task(2,2,0,2), (Task(2,4,1,2))]])
#print(sol)
#print(sol.is_realisable())
#print(sol.get_duration())

results = js.heuristique_gloutonne(strategy='EST_LRPT')
print(results)
print(results.gant)
print(results.duration)
