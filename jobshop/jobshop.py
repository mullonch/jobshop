from __future__ import annotations

import itertools
from typing import List, Dict
from functools import cached_property
from copy import deepcopy

from jobshop.graph import Graph
from jobshop.utils import str_table

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
        return f"{self.nodename} ({str(self.machine)})"

    def __repr__(self):
        return self.nodename

    @property
    def nodename(self):
        return f"st({self.ijob}, {self.itask})"

class Solution(Graph):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    @cached_property
    def starts(self)->Dict[str, float]:
        if self.has_cycle():
            return dict.fromkeys(self.v, float('inf'))
        starts = dict.fromkeys(self.v, 0.)
        for node in self.topological_list():
            starts[node] = max([starts[node]] + [starts[p] + self.get_cost(p, node) for p in self.incomings(node)])
        return starts
    
    @cached_property
    def critical_path(self):
        crit_orig = dict.fromkeys(self.v, "-")
        starts = dict.fromkeys(self.v, 0.)
        for ne in self.neighbors("stS"):
            crit_orig[ne] = "stS"

        for node in self.topological_list():
            for p in self.incomings(node):
                if starts[p] + self.get_cost(p, node) > starts[node]:
                    starts[node] = starts[p] + self.get_cost(p, node)
                    crit_orig[node] = p
        result = ['stF']
        while result[0] != 'stS':
            result = [crit_orig[result[0]]] + result
        return result
    
    @cached_property 
    def invertible_tasks(self):
        tasks = [self.problem.get_task_by_nodename(node) for node in self.critical_path[1:-1]]
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
    
    @staticmethod
    def from_ressource_matrix(problem, matrix)->Solution:
        """
        :param problem: probleme de Jobshop
        :param matrix: solution au problème donné sous la représentation en ressources (liste de liste de Task)
        :return: Solution au problème donné (objet Solution)
        """
        s = Solution(problem)
        s += problem.time_graph
        for machine in matrix:
            for i in range(1, len(machine)):
                s.link(machine[i - 1].nodename, machine[i].nodename, cost=machine[i - 1].duration)
        return s

    def __str__(self):
        return str(self.starts)

    def is_realisable(self):
        return not self.has_cycle()
    
    def task_start(self, task_name):
        return self.starts[task_name]

    @property
    def matrix(self):
        return [[self.starts[task.nodename] for task in job] for job in self.problem.jobs]

    @property
    def str_matrix(self):
        return str_table([list(range(self.problem.nb_machines))] + self.matrix,
                     ["num_colonne"] + ["Job " + str(i) for i in range(self.problem.nb_jobs)])

    @property
    def ressource_matrix(self):
        # TODO optimiser cette merde
        res = []
        for imac in range(self.problem.nb_machines):
            tasks = [task.nodename for task in self.problem.get_tasks_by_machine(imac)]
            tasks.sort(key=lambda x: self.starts[x])
            res.append(tasks)
        return res

    @property
    def str_ressource_matrix(self):
        return str_table(self.ressource_matrix, ["r" + str(i) for i in range(self.problem.nb_machines)])

    @property
    def job_matrix(self):
        # TODO : revoir ça (ça marche mais c'est moyen)
        # TODO : rentre cet algorithme stable (de préférence)
        tasks = [task for task in self.v if task != "stS" and task != "stF"]
        tasks.sort(key=lambda x: self.starts[x])
        return [t[3] for t in tasks]

    @property
    def str_job_matrix(self):
        return str_table([self.job_matrix], ["num job"])

    @property
    def gant(self):
        # TODO : Optimiser cette merde dégueulasse
        res = "DIAGRAMME DE GANT : \n"
        for imac in range(self.problem.nb_machines):
            tbm = [[(task, self.starts[task.nodename]) for task in job if task.machine == imac][0] for job in
                   self.problem.jobs]
            tbm.sort(key=lambda x: x[1])
            str_machine = "Mac n°" + str(imac) + " : "
            actual_position = 0
            for task in tbm:
                diff = int(task[1] - actual_position)
                str_machine += diff * "___"
                str_machine += ("[" + str(task[0].ijob) + "]") * task[0].duration
                actual_position += task[0].duration + diff
            res += str_machine + "\n"
        return res
    
    def get_cost(self, node_from, node_to=None)->float:
        if node_to is not None:
            return super().get_cost(node_from, node_to)
        for a in self.e:
            if a.node_from == str(node_from):
                return a.cost
        return float("inf")

    def longest_path_length(self, node_from, node_to):
        # TODO : refaire ça différement (bruteforce dégueulasse)
        if self.has_cycle():
            return float("inf")
        if node_from == node_to:
            return 0
        chemins_possibles = []
        for next_node in self.neighbors(node_from):
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
        return self.starts["stF"]

    def solution_neighbors(self, forbidden=[]):
        res = dict()
        invertibles = self.invertible_tasks
        for permutation in invertibles:
            ipermutables = [(i - 1, i) for i in range(1, len(permutation))]
            for i1, i2 in ipermutables:
                if (permutation[i1], permutation[i2]) not in forbidden:
                    s = deepcopy(self)
                    s.link(permutation[i2].nodename, permutation[i1].nodename, cost=self.get_cost(permutation[i2]))
                    if i1 > 0:
                        s.link(permutation[i1 - 1].nodename, permutation[i2].nodename,
                               cost=self.get_cost(permutation[i1 - 1]))
                    if i2 < len(permutation) - 1:
                        s.link(permutation[i1].nodename, permutation[i2 + 1].nodename,
                               cost=self.get_cost(permutation[i1]))
                    s.unlink(*[p.nodename for p in permutation[max(i1 - 1, 0):min(i2, len(permutation) - 1) + 2]])
                    res[s] = (permutation[i1].nodename, permutation[i2].nodename)
        return res

class JobShop:
    def __init__(self, nb_jobs, nb_machines, jobs):
        self.nb_jobs = nb_jobs
        self.nb_machines = nb_machines
        self.jobs = jobs

    def add_job(self, job:Task):
        self.jobs.append(job)

    @staticmethod
    def from_instance_file(filename:str="instances/default_instance")->JobShop:
        with open(filename, "r") as file:
            lines  = [[int(a) for a in filter(None, line.split(" "))] for line in file.read().split("\n") if line != "" and line[0] != "#"]
        nb_jobs, nb_machines = lines.pop(0)
        
        # TODO : améliorer les 6 lignes ci-dessous (possibles de le faire sans stockage du res. de la dissociation des lignes)
        machines = [[job[i] for i in range(len(job)) if not i % 2] for job in lines]
        durations = [[job[i] for i in range(len(job)) if i % 2] for job in lines]
        jobs = []
        for j in range(nb_jobs):
            jobs.append([Task(machines[j][i], durations[j][i], j, i) for i in range(nb_machines)])
        return JobShop(nb_jobs, nb_machines, jobs)
    
    @property
    def naive_upper_bound(self):
        return sum(sum(task.duration for task in job) for job in self.jobs)
    
    @property 
    def naive_lower_bound(self):
        return max(sum(task.duration for task in job) for job in self.jobs)

    @property
    def naive_bounds(self):
        return self.naive_lower_bound, self.naive_upper_bound
    
    def get_tasks_by_machine(self, machine: int):
        return [[task for task in job if task.machine == machine][0] for job in self.jobs]

    def get_job(self, id_job: int):
        return self.jobs[id_job]

    def get_task(self, id_job: int, no_task: int):
        return self.jobs[id_job][no_task]

    def get_task_by_nodename(self, nodename):
        infos = nodename.split(", ")
        return self.get_task(int(infos[0][3:]), int(infos[1][:-1]))

    @property
    def time_graph(self)->Graph:
        res = Graph("stS", "stF")
        for job in self.jobs:
            prec_node = "stS"#= ("stS", 0)
            prec_cost = 0
            for task in job:
                res += task.nodename
                res.link(prec_node, task.nodename, cost=prec_cost)
                prec_node = task.nodename
                prec_cost = task.duration
            res.link(prec_node, "stF", cost=prec_cost)
        return res
    
    @property
    def ressources_graphs(self)->List[Graph]:
        res = [Graph() for _ in range(self.nb_machines)]
        for imac in range(self.nb_machines):
            tasks = self.get_tasks_by_machine(imac)
            res[imac] += [task.nodename for task in tasks]
            for t1, t2 in itertools.combinations(tasks, 2):
                res[imac].link(t1.nodename, t2.nodename, cost=t1.duration)
                res[imac].link(t2.nodename, t1.nodename, cost=t2.duration)
        return res
    
    @property
    def graph(self)->Graph:
        return self.time_graph + self.ressources_graphs