from utils.JobShop import *

js = JobShop()
sol = js.pick_mf_solution()

results = js.heuristique_gloutonne()
print(results)


"""
if sol.is_realisable():
    print("Total duration : ", sol.get_duration())
    print(sol.gant())
    print(sol.str_matrix())
    print(sol.str_ressource_matrix())
    print(sol.str_job_matrix())

"""