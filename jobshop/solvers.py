import random
from copy import deepcopy
from abc import abstractmethod, ABCMeta

from jobshop.jobshop import JobShop, Solution
class SolverInterface(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def solve(self, problem:JobShop)->Solution:
        ...

class GreedySolver(SolverInterface):
    ACCEPTED_STRATEGIES = ["SPT", "LPT", "SRPT", "LRPT", "EST_SPT", "EST_LPT", "EST_LRPT", "EST_SRPT", "RANDOM"]

    def __init__(self, strategy="SPT", p_random=0):
        self.trace = []
        if strategy.upper() not in self.ACCEPTED_STRATEGIES:
            raise NameError("Unrecognized strategy : " + strategy)
        self.strategy = strategy
        self.p_random = p_random

    def solve(self, problem: JobShop) -> Solution:
        result = [[] for _ in range(problem.nb_machines)]
        jobs = deepcopy(problem.jobs)
        realisables = [job[0] for job in jobs]

        def EST(x):
            start_dates = [0 for _ in range(len(x))]
            possible_next_tasks = []

            for ind, task in enumerate(x):
                start_dates[ind] = max(EST.djob[task.ijob], EST.dmac[task.machine])
            for ind, task in enumerate(x):
                if start_dates[ind] == min(start_dates):
                    possible_next_tasks += [task]

            next_task = selectors[self.strategy[4:]](possible_next_tasks)
            end = next_task.duration + max(EST.djob[next_task.ijob], EST.dmac[next_task.machine])

            EST.dmac[next_task.machine] = end
            EST.djob[next_task.ijob] = end
            return next_task

        EST.dmac = [0 for _ in range(problem.nb_machines)]
        EST.djob = [0 for _ in range(problem.nb_jobs)]
        selectors = {
            "SPT": lambda x: next(a for a in x if all(a.duration <= b.duration for b in x)),
            "LPT": lambda x: next(a for a in x if all(a.duration >= b.duration for b in x)),
            "SRPT": lambda x: next(a for a in x if all(
                sum(t.duration for t in jobs[a.ijob]) <= sum(t.duration for t in jobs[b.ijob]) for b in x)),
            "LRPT": lambda x: next(a for a in x if all(
                sum(t.duration for t in jobs[a.ijob]) >= sum(t.duration for t in jobs[b.ijob]) for b in x)),
            "EST_": EST,
            "rand": lambda x: random.choice(x)
        }

        while len(realisables) > 0:
            if random.random() < self.p_random:
                next_task = selectors["rand"](realisables)
            else:
                next_task = selectors[self.strategy[:4]](realisables)
            result[next_task.machine].append(jobs[next_task.ijob].pop(0))
            realisables = [job[0] for job in jobs if len(job) > 0]
        return Solution.from_ressource_matrix(problem, result)


class DescenteSolver(SolverInterface):
    def __init__(self):
        ...

    def solve(self, problem: JobShop, start=None):
        if start is None:
            start = GreedySolver("EST_LRPT").solve(problem)
        for neighbor in start.solution_neighbors():
            if neighbor.duration < start.duration:
                return self.solve(problem, neighbor)
        return start


class RandomSolver(SolverInterface):
    def __init__(self):
        ...
        
    def solve(self, problem: JobShop):
        return GreedySolver("RANDOM").solve(problem)


class MultipleDescenteSolver(SolverInterface):
    def __init__(self, nb_starters=10, starter_strategy="random", starter_randomisation=1):
        self.nb_starters = nb_starters
        self.starter_strategy = starter_strategy
        self.starter_randomisation = starter_randomisation

    def solve(self, problem: JobShop):
        return min([DescenteSolver().solve(problem, start=GreedySolver(strategy=self.starter_strategy,
                                                                       p_random=self.starter_randomisation).solve(
            problem)) for _ in range(self.nb_starters)], key=lambda x: x.duration)
