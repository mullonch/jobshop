from utils.JobShop import *

js = JobShop()

sol = js.pick_default_solution()
toto = sol.inv_blocks_of_critical_path()
print(toto)
