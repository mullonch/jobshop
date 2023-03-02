
from jobshop.jobshop import JobShop
from jobshop.solvers import GreedySolver

js = JobShop.from_instance_file("instances/ft10")
solution = GreedySolver("SPT").solve(js)

print("makespan pour ft10 : ")
print(solution.longest_path_length)
