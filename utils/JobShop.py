"""
    Module permettant de matérialiser des problèmes de JobShop
"""
from copy import deepcopy
from utils.Graphe import *


class Task:
    """
        Structure permettant de représenter un noeuf de graphe de JobShop
    """
    def __init__(self, machine, duration, ijob, itask):
        self.machine = machine
        self.duration = duration
        self.ijob = ijob
        self.itask = itask

    def __repr__(self):
        return self.nodename

    @property
    def nodename(self):
        return "st(" + str(self.ijob) + ", " + str(self.itask) + ")"

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
            self.jobs.append([Task(machines[j][i] - 1, durations[j][i], j, i) for i in range(self.nb_machines)])

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


    def pick_first_solution(self):
        # TODO faire mieux que ça
        solution = Solution(self)
        solution += self.get_time_graphe()
        for g in self.get_ressources_graphes():
            solution += next(iter(g.E))
        return solution

    def pick_default_solution(self):
        """
        :return: pour le pb par défaut, renvoie la solution vue en cours (makespan 12)
        """
        solution = Solution(self)
        solution += self.get_time_graphe()
        solution.link(("st(0, 0)", "st(1, 1)"), 3)
        solution.link(("st(1, 0)", "st(0, 1)"), 2)
        solution.link(("st(0, 2)", "st(1, 2)"), 2)
        return solution

    def heuristique_gloutonne(self, strategy="SPT"):
        """
        :param strategy: chaine de caractère définissant la stratégie de choix :
        SPT : Shortest Processing Time
        LPT : Longest Processing Time
        SRPT : Shortest Remaining Processing Time
        LRPT : Longest Remaining Processing Time
        :return:
        """
        result = [[] for _ in range(self.nb_machines)]
        jobs = deepcopy(self.jobs)
        realisables = [job[0] for job in jobs]

        dmac = [0 for _ in range(self.nb_machines)]
        djob = [0 for _ in range(self.nb_jobs)]

        def EST_STP(realisables):

            start_dates =[0 for _ in range(len(realisables))]

            possible_next_tasks = []

            for ind,task in enumerate(realisables):
                start_dates[ind] = max(djob[task.ijob],dmac[task.machine])


            for ind,task in enumerate(realisables):
                if start_dates[ind] == min(start_dates):
                    possible_next_tasks+=[task]



            if len(possible_next_tasks)>1:
                next_task = selectors[strategy[4:]](possible_next_tasks)
            else:
                next_task = possible_next_tasks[0]

            return next_task

        selectors = {
            "SPT": lambda x: next(a for a in x if a.duration == min(a.duration for a in x)),
            "LPT": lambda x: next(a for a in x if a.duration == max(a.duration for a in x)),
            "SRPT": lambda x: next(a for a in x if sum(t.duration for t in jobs[a.ijob]) == min(
                sum(a.duration for a in j) for j in jobs if len(j) > 0)),
            "LRPT": lambda x: next(a for a in x if sum(t.duration for t in jobs[a.ijob]) == max(
                sum(a.duration for a in j) for j in jobs if len(j) > 0))
        }


        while len(realisables) > 0:

            if strategy[:4] == 'EST_':
                next_task = EST_STP(realisables)


                end = next_task.duration+max(djob[next_task.ijob],dmac[next_task.machine])
                dmac[next_task.machine] = end
                djob[next_task.ijob] = end

            else:
                next_task = selectors[strategy](realisables)

            result[next_task.machine].append(jobs[next_task.ijob].pop(0))
            realisables = [job[0] for job in jobs if len(job) > 0]



        return Solution.from_ressource_matrix(self, result)

class Solution(Graphe):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def init_starts(self):
        self.starts = dict.fromkeys(self.V, 0)
        for node in self.topological_list():
            self.starts[node] = max(
                [self.starts[node]] + [self.starts[p] + self.get_cost(p, node) for p in self.get_incomings(node)])

    def blocks_of_critical_path(self):
        pass

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
        :return: Un booléen iniquant si la solution est réalisable (le graphe ne contient pas de cycle)
        """
        return not self.has_cycle()

    @property
    def matrix(self):
        return [[self.date_debut_tache(task.nodename) for task in job] for job in self.problem.jobs]

    @property
    def str_matrix(self):
        return table([list(range(self.problem.nb_machines))] + self.matrix(), ["num_colonne"] + ["Job " + str(i) for i in range(self.problem.nb_jobs)])

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
        if self.starts[task_name] is None:
            self.init_starts()
            # self.starts[task_name] = self.longest_path_length("stS", task_name)
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
    def duration(self):
        return self.date_debut_tache("stF")


def table(tab, lig_names):
    l_first_col = max(len(a) for a in lig_names)
    l_col = max(len(str(a)) for ligne in tab for a in ligne)
    c = len(tab[0])
    res = "╔" + "═" * (l_first_col + 2) + ("╤══" + "═" * l_col) * c + "╗\n" + "".join(
        ("║{:^" + str(l_first_col + 2) + "}" + "".join("│{:^" + str(l_col + 2) + "}" for _ in range(c)) + "║\n").format(
            *[lig_names[ligne]] + [str(a) for a in tab[ligne]]) for ligne in range(len(tab))) + "╚" + "═" * (
                  l_first_col + 2) + ("╧══" + "═" * l_col) * c + "╝"
    return res
