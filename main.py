from jobshop.jobshop import JobShop
from jobshop.solvers import GreedySolver, RandomSearchSolver, RandomSolver


js = JobShop.from_instance_file(filename="./instances/abz5")
print(js.naive_bounds) #(859, 7773)
solver = GreedySolver("EST_SPT")
solver2 = RandomSearchSolver(budget=50)
solution = solver.solve(js)
solution2 = solver2.solve(js)
print(solution.duration) #1352
print(solution2.duration)