from utils.JobShop import *

js = JobShop()
sol = js.pick_mf_solution()

if sol.is_realisable():
    print("Total duration : ", sol.get_duration())
    print(sol.gant())
    print(sol.str_matrix())
    print(sol.str_ressource_matrix())
    print(sol.str_job_matrix())
