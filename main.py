from utils.JobShop import *
from datetime import timedelta
js = JobShop()


sol = js.descente()
print(sol.duration)

