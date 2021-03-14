"""
    Module permettant de matérialiser des problèmes de JobShop
"""
import itertools
from copy import deepcopy
from datetime import datetime, timedelta
import random
from utils2.Graphe import Graphe

class SolverInterface:
    def __init__(self):
        raise NotImplementedError
    def solve(self, problem):
        raise NotImplementedError


class Task:
    """
        Structure permettant de représenter un noeuf de graphe de JobShop
    """

    def __init__(self, machine, duration, ijob, itask):
        self.machine = machine
        self.duration = duration
        self.ijob = ijob
        self.itask = itask

    def temprepr(self):
        return self.nodename + "(" + str(self.machine) + ")"

    def __repr__(self):
        return self.nodename

    @property
    def nodename(self):
        return "st(" + str(self.ijob) + ", " + str(self.itask) + ")"



class Solution(Graphe):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def init_starts(self):
        if hasattr(self, 'starts'):
            return
        if self.has_cycle():
            self.starts = dict.fromkeys(self.V, float('inf'))
        else:
            self.starts = dict.fromkeys(self.V, 0)
            for node in self.topological_list():
                self.starts[node] = max(
                    [self.starts[node]] + [self.starts[p] + self.get_cost(p, node) for p in self.get_incomings(node)])

    def blocks_of_critical_path(self):
        crit_orig = dict.fromkeys(self.V, 0)
        starts = dict.fromkeys(self.V, 0)
        for ne in self.get_neighbors("stS"):
            crit_orig[ne] = "stS"

        for node in self.topological_list():
            for p in self.get_incomings(node):
                if starts[p] + self.get_cost(p, node) > starts[node]:
                    starts[node] = starts[p] + self.get_cost(p, node)
                    crit_orig[node] = p
        result = ['stF']
        while result[0] != 'stS':
            result = [crit_orig[result[0]]] + result
        return result

    def get_invertibles(self):
        tasks = [self.problem.get_task_by_nodename(node) for node in self.blocks_of_critical_path()[1:-1]]
        res = []
        current_machine = -1
        current_list = []
        for t in tasks:
            if t.machine != current_machine:
                if len(current_list) > 1:
                    res += [current_list]
                current_list = [t]
            else:
                current_list.append(t)
            current_machine = t.machine
        if len(current_list) > 1:
            res += [current_list]
        return res
        # return [(tasks[i-1], tasks[i]) for i in range(1, len(tasks)) if tasks[i-1].machine == tasks[i].machine]

    @staticmethod
    def from_ressource_matrix(problem, matrix):
        """
        :param problem: probleme de Jobshop
        :param matrix: solution au problème donné sous la représentation en ressources (liste de liste de Task)
        :return: Solution au problème donné (objet Solution)
        """
        s = Solution(problem)
        s += problem.get_time_graphe()
        for machine in matrix:
            for i in range(1, len(machine)):
                s.link((machine[i - 1].nodename, machine[i].nodename), machine[i - 1].duration)
        return s

    def __str__(self):
        if not hasattr(self, 'starts'):
            self.init_starts()
        return str(self.starts)

    def is_realisable(self):
        """
        :return: Un booléen indiquant si la solution est réalisable (le graphe ne contient pas de cycle)
        """
        return not self.has_cycle()

    @property
    def matrix(self):
        return [[self.date_debut_tache(task.nodename) for task in job] for job in self.problem.jobs]

    @property
    def str_matrix(self):
        return table([list(range(self.problem.nb_machines))] + self.matrix,
                     ["num_colonne"] + ["Job " + str(i) for i in range(self.problem.nb_jobs)])

    @property
    def ressource_matrix(self):
        # TODO optimiser cette merde
        res = []
        for imac in range(self.problem.nb_machines):
            tasks = [task.nodename for task in self.problem.get_tasks_by_machine(imac)]
            tasks.sort(key=lambda x: self.date_debut_tache(x))
            res.append(tasks)
        return res

    @property
    def str_ressource_matrix(self):
        return table(self.ressource_matrix, ["r" + str(i) for i in range(self.problem.nb_machines)])

    @property
    def job_matrix(self):
        # TODO : revoir ça (ça marche mais c'est moyen)
        # TODO : rentre cet algorithme stable (de préférence)
        tasks = [task for task in self.V if task != "stS" and task != "stF"]
        tasks.sort(key=lambda x: self.date_debut_tache(x))
        return [t[3] for t in tasks]

    @property
    def str_job_matrix(self):
        return table([self.job_matrix], ["num job"])

    @property
    def gant(self):
        # TODO : Optimiser cette merde dégueulasse
        res = "DIAGRAMME DE GANT : \n"
        for imac in range(self.problem.nb_machines):
            tbm = [[(task, self.date_debut_tache(task.nodename)) for task in job if task.machine == imac][0] for job in
                   self.problem.jobs]
            tbm.sort(key=lambda x: x[1])
            str_machine = "Mac n°" + str(imac) + " : "
            actual_position = 0
            for task in tbm:
                diff = task[1] - actual_position
                str_machine += diff * "___"
                str_machine += ("[" + str(task[0].ijob) + "]") * task[0].duration
                actual_position += task[0].duration + diff
            res += str_machine + "\n"
        return res

    def date_debut_tache(self, task_name):
        self.init_starts()
        return self.starts[task_name]

    def longest_path_length(self, node_from, node_to):
        # TODO : refaire ça différement (bruteforce dégueulasse)
        if self.has_cycle():
            return float("inf")
        if node_from == node_to:
            return 0
        chemins_possibles = []
        for next_node in self.get_neighbors(node_from):
            chemins_possibles += [self.get_cost(node_from, next_node) + self.longest_path_length(next_node, node_to)]
        if len(chemins_possibles) == 0:
            return -10000
        else:
            return max(chemins_possibles)

    @property
    def gant_duration(self):
        return max(len(e.split(" : ")[1]) // 3 for e in self.gant.split("\n")[1:-1])

    @property
    def duration(self):
        return self.date_debut_tache("stF")

    def get_cost(self, node_from, node_to=None):
        if node_to is not None:
            return super().get_cost(node_from, node_to)
        else:
            for a in self.E:
                if a.node_from == str(node_from):
                    return a.cost

    def solution_neighbors(self, forbidden=[]):
        res = dict()
        invertibles = self.get_invertibles()
        for permutation in invertibles:
            ipermutables = [(i - 1, i) for i in range(1, len(permutation))]
            for i1, i2 in ipermutables:
                if (permutation[i1], permutation[i2]) not in forbidden:
                    s = deepcopy(self)
                    s.link((permutation[i2].nodename, permutation[i1].nodename), cost=self.get_cost(permutation[i2]))
                    if i1 > 0:
                        s.link((permutation[i1 - 1].nodename, permutation[i2].nodename),
                               cost=self.get_cost(permutation[i1 - 1]))
                    if i2 < len(permutation) - 1:
                        s.link((permutation[i1].nodename, permutation[i2 + 1].nodename),
                               cost=self.get_cost(permutation[i1]))
                    s.unlink([p.nodename for p in permutation[max(i1 - 1, 0):min(i2, len(permutation) - 1) + 2]])
                    s.init_starts()
                    res[s] = (permutation[i1].nodename, permutation[i2].nodename)

        return res


class JobShop:
    """
        Classe représentant un JobShop
            -> Un JobShop est un ensemble de jobs
            -> Chaque job est une liste de taches
            -> Chaque tache a une durée et une machine associée.
    """

    def __init__(self, filename="instances/default_instance"):
        """
        :param filename: fichier de données de jobShop (voir répertoire instances)
        """
        lines = [[int(a) for a in filter(None, line.split(" "))] for line in open(filename, "r").read().split("\n") if
                 line != "" and line[0] != "#"]
        self.nb_jobs, self.nb_machines = lines.pop(0)
        # TODO : améliorer les 6 lignes ci-dessous (possibles de le faire sans stockage du res. de la dissociation des lignes)
        machines = [[job[i] for i in range(len(job)) if not i % 2] for job in lines]
        durations = [[job[i] for i in range(len(job)) if i % 2] for job in lines]
        self.jobs = []
        for j in range(self.nb_jobs):
            self.jobs.append([Task(machines[j][i], durations[j][i], j, i) for i in range(self.nb_machines)])

    def get_naive_upper_bound(self):
        """
        :return: Une borne supérieure naive pour le makespan des solutions de ce jobShop (some des durées des tâches)
        """
        return sum(sum(task.duration for task in job) for job in self.jobs)

    def get_naive_lower_bound(self):
        """
        :return: Une borne inférieure naive pour le makespan des solutions de ce jobShop
        """
        return max(sum(task.duration for task in job) for job in self.jobs)

    def get_naive_bounds(self):
        """
        :return: Un encadrement naïf du makespan de la solution
        """
        return self.get_naive_lower_bound(), self.get_naive_upper_bound()

    def get_tasks_by_machine(self, machine: int):
        """
        :param machine: numéro de machine
        :return: Toutes les tâches devant passer sur cette machine
        """
        return [[task for task in job if task.machine == machine][0] for job in self.jobs]

    def get_job(self, id_job: int):
        """
        :param id_job: identifiant de job
        :return: la liste des taches de ce job
        """
        return self.jobs[id_job]

    def get_task(self, id_job: int, no_task: int):
        """
        :param id_job: identifiant de job
        :param no_task: numéro de tâche pour un job donné
        :return: no_task ième tache du id_job ième job
        """
        return self.jobs[id_job][no_task]

    def get_task_by_nodename(self, nodename):
        infos = nodename.split(", ")
        return self.get_task(int(infos[0][3:]), int(infos[1][:-1]))

    def get_time_graphe(self):
        res = Graphe()
        res += ["stS", "stF"]
        for job in self.jobs:
            prec = ("stS", 0)
            for task in job:
                res += task.nodename
                res.link((prec[0], task.nodename), prec[1])
                prec = (task.nodename, task.duration)
            res.link((prec[0], "stF"), prec[1])
        return res

    def get_graphe(self):
        """
        :return: Le graphe complet représentant le problème avec toutes ses contraintes.
        """
        return self.get_time_graphe() + self.get_ressources_graphes()

    def get_ressources_graphes(self):
        """
        :return: une liste de graphes sommables représentant les différentes contraintes de partage des ressources
        """
        res = [Graphe() for _ in range(self.nb_machines)]
        for imac in range(self.nb_machines):
            tasks = self.get_tasks_by_machine(imac)
            res[imac] += [task.nodename for task in tasks]
            for t1, t2 in itertools.combinations(tasks, 2):
                res[imac].link((t1.nodename, t2.nodename), cost=t1.duration)
                res[imac].link((t2.nodename, t1.nodename), cost=t2.duration)
        return res

    def solve(self, solver : SolverInterface) ->Solution:
        return solver.solve(self)


class GreedySolver(SolverInterface):
    accepted_strategies = ["SPT", "LPT", "SRPT", "LRPT", "EST_SPT", "EST_LPT", "EST_LRPT", "EST_SRPT", "random"]

    def __init__(self, strategy="SPT", p_random=0):
        self.trace = []
        if strategy not in self.accepted_strategies:
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
    def solve(self, problem: JobShop, start=None):
        if start is None:
            start = GreedySolver("EST_LRPT").solve(problem)
        for neighbor in start.solution_neighbors():
            if neighbor.duration < start.duration:
                return self.solve(problem, neighbor)
        return start


class RandomSolver():
    def solve(self, problem: JobShop):
        return GreedySolver("random").solve(problem)


class MultipleDescenteSolver(SolverInterface):
    def __init__(self, nb_starters=10, starter_strategy="random", starter_randomisation=100):
        self.nb_starters = nb_starters
        self.starter_strategy = starter_strategy
        self.starter_randomisation = starter_randomisation

    def solve(self, problem: JobShop):
        return min([DescenteSolver().solve(problem, start=GreedySolver(strategy=self.starter_strategy,
                                                                       p_random=self.starter_randomisation).solve(
            problem)) for _ in range(self.nb_starters)], key=lambda x: x.duration)


class TabooSolver(SolverInterface):
    def __init__(self, max_iter=25, timeout=999999999, return_time=5):
        self.max_iter = max_iter
        self.timeout = timeout
        self.return_time = return_time

    def solve(self, problem):
        s_init = GreedySolver("EST_LRPT").solve(problem)
        # Initialiser la matrice sTaboo:
        nodes = s_init.V
        forbidden = []
        sTaboo_matrix = {}
        for v1 in nodes:
            sTaboo_matrix[v1] = {}
            for v2 in nodes:
                sTaboo_matrix[v1][v2] = 0

        # mémoriser la meilleure solution
        best = s_init

        # solution courrante
        s = s_init

        # solutions tabou
        sTaboo = [s]

        # compteur d'iétrations
        k = 0
        tinit = datetime.now()

        # Exploration des voisinages successifs
        while k < self.max_iter and (datetime.now() - tinit) < timedelta(seconds=self.timeout):
            # remplir la liste de permutations interdite à partir de sTaboo_matrix:
            for v1 in nodes:
                for v2 in nodes:
                    if sTaboo_matrix[v1][v2] > k:
                        forbidden += [
                            (problem.get_task_by_nodename(nodename=v1), problem.get_task_by_nodename(nodename=v2))]
                    if 0 < sTaboo_matrix[v1][v2] <= k:
                        if (
                                problem.get_task_by_nodename(nodename=v1),
                                problem.get_task_by_nodename(nodename=v2)) in forbidden:
                            forbidden.remove(
                                (problem.get_task_by_nodename(nodename=v1), problem.get_task_by_nodename(nodename=v2)))

            # choose the best neighbor s_prime which is not tabou
            neighbors = s.solution_neighbors(forbidden)
            obj = float('inf')

            if len(neighbors) == 0:
                break
            else:
                for n in neighbors:
                    # print(neighbors[n])
                    # print('n.duration : ',n.duration)
                    if n.duration <= obj:
                        obj = n.duration
                        sprime = n
            sTaboo_matrix[neighbors[sprime][0]][neighbors[sprime][1]] = k + self.return_time
            s = sprime

            if sprime.duration < best.duration:
                best = sprime
            k += 1
        return best





def table(tab, lig_names):
    l_first_col = max(len(a) for a in lig_names)
    l_col = max(len(str(a)) for ligne in tab for a in ligne)
    c = len(tab[0])
    res = "╔" + "═" * (l_first_col + 2) + ("╤══" + "═" * l_col) * c + "╗\n" + "".join(
        ("║{:^" + str(l_first_col + 2) + "}" + "".join("│{:^" + str(l_col + 2) + "}" for _ in range(c)) + "║\n").format(
            *[lig_names[ligne]] + [str(a) for a in tab[ligne]]) for ligne in range(len(tab))) + "╚" + "═" * (
                  l_first_col + 2) + ("╧══" + "═" * l_col) * c + "╝"
    return res
