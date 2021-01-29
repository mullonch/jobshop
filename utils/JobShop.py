"""
    Module permettant de matérialiser des problèmes de JobShop
"""
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
        self.node_name = Task.node_name(ijob, itask)

    def __repr__(self):
        return self.node_name
        #return "{job:"+str(self.ijob)+","+"task:"+str(self.itask)+","+"m:" + str(self.machine) + ", d:" + str(self.duration) + "}"

    @staticmethod
    def node_name(ijob, itask):

        """
        :param ijob: identifiant du job dans lequel est la tâche
        :param itask: position de la tache dans le job (0 = première tache)
        :return: nom du noeud du graphe correspondant à la tache
        """
        return "st(" + str(ijob) + ", " + str(itask) + ")"


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
        machines = [[task[i] for i in range(len(task)) if not i % 2] for task in lines]
        durations = [[task[i] for i in range(len(task)) if i % 2] for task in lines]
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
        """
        :return: Un objet Graphe représentant les contraintes temporelles du problème.
        """
        g = Graphe()
        g += ["stS", "stF"]
        for ijob in range(self.nb_jobs):
            prec = "stF"
            for itask in range(self.nb_machines, 0, -1):
                node_name = Task.node_name(ijob, itask - 1)
                g += node_name
                g.link((node_name, prec), cost=self.get_task(ijob, itask - 1).duration)
                prec = node_name
            g.link(("stS", prec))
        return g

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
            res[imac] += [task.node_name for task in tasks]
            for t1, t2 in itertools.combinations(tasks, 2):
                res[imac].link((t1.node_name, t2.node_name), cost=t1.duration)
                res[imac].link((t2.node_name, t1.node_name), cost=t2.duration)
        return res

    def pick_first_solution(self):
        # TODO faire mieux que ça
        solution = Solution(self)
        solution += self.get_time_graphe()
        for g in self.get_ressources_graphes():
            solution += next(iter(g.E))
        return solution

    def pick_mf_solution(self):
        """
        :return: pour le pb par défaut, renvoie la solution vue en cours (makespan 12)
        """
        solution = Solution(self)
        solution += self.get_time_graphe()
        solution.link(("st(0, 0)", "st(1, 1)"), 3)
        solution.link(("st(1, 0)", "st(0, 1)"), 2)
        solution.link(("st(0, 2)", "st(1, 2)"), 2)
        return solution

    def pick_solution(self):
        # TODO : faire un selecteur de solution potable
        return self.pick_first_solution()

    def heuristique_gloutonne(self,priority = 'SPT'):

        # Init : Determiner l'ensemble des tâches réalisables
        task_per_mac=[]
        for i in range(self.nb_machines):
            task_per_mac +=[[]]
        print(task_per_mac)
        task_list = []
        for job in self.jobs:
            task_list += [job[0]]

        # boucle

        if priority == 'SPT':

            while len(task_list) != 0:
                duree = float('inf')
                for ind,t in enumerate(task_list):
                    if t.duration < duree:
                        duree = t.duration
                        next_task = t
                        ind_pop=ind

                mac = next_task.machine
                task_per_mac[mac].append(next_task)
                task_list.pop(ind_pop)

                j = next_task.ijob
                i = next_task.itask
                if i < len(self.jobs[j])-1:
                    task_list.append(self.jobs[j][i+1])

        return task_per_mac














class Solution(Graphe):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.starts = dict.fromkeys([task.node_name for job in self.problem.jobs for task in job])

    def is_realisable(self):
        """
        :return: Un booléen iniquant si la solution est réalisable (le graphe ne contient pas de cycle)
        """
        return not self.has_cycle()

    def representation(self, mode):
        if mode.lower == "gant":
            return self.gant()
        if mode.lower == "matrix":
            return self.str_matrix()
        if mode.lower == "ressources":
            return self.str_ressource_matrix()
        if mode.lower == "job":
            return self.str_job_matrix()

    def matrix(self):
        return [[self.date_debut_tache(Task.node_name(ijob, itask)) for itask in range(self.problem.nb_machines)] for
                ijob in range(self.problem.nb_jobs)]

    def str_matrix(self):
        tab = [list(range(self.problem.nb_machines))] + self.matrix()
        fcol = ["num_colonne"] + ["Job " + str(i) for i in range(self.problem.nb_jobs)]
        return table(tab, fcol)

    def ressource_matrix(self):
        # TODO optimiser cette merde
        res = []
        for imac in range(self.problem.nb_machines):
            tasks = [task.node_name for task in self.problem.get_tasks_by_machine(imac)]
            tasks.sort(key=lambda x: self.date_debut_tache(x))
            res.append(tasks)
        return res

    def str_ressource_matrix(self):
        return table(self.ressource_matrix(), ["r" + str(i) for i in range(self.problem.nb_machines)])

    def job_matrix(self):
        # TODO : revoir ça (ça marche mais c'est moyen)
        # TODO : rentre cet algorithme stable (de préférence)
        tasks = [task for task in self.V if task != "stS" and task != "stF"]
        tasks.sort(key=lambda x: self.date_debut_tache(x))
        return [t[3] for t in tasks]

    def str_job_matrix(self):
        return table([self.job_matrix()], ["num job"])

    def gant(self):
        # TODO : Optimiser cette merde dégueulasse
        res = "DIAGRAMME DE GANT : \n"
        for imac in range(self.problem.nb_machines):
            tbm = [[(task, self.date_debut_tache(task.node_name)) for task in job if task.machine == imac][0] for job in
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
            self.starts[task_name] = self.longest_path_length("stS", task_name)
        return self.starts[task_name]

    def longest_path_length(self, node_from, node_to):
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

    def get_duration(self):
        return self.longest_path_length("stS", "stF")


def table(tab, lig_names):
    l_first_col = max(len(a) for a in lig_names)
    l_col = max(len(str(a)) for ligne in tab for a in ligne)
    c = len(tab[0])
    res = "╔" + "═" * (l_first_col + 2) + ("╤══" + "═" * l_col) * c + "╗\n" + "".join(
        ("║{:^" + str(l_first_col + 2) + "}" + "".join("│{:^" + str(l_col + 2) + "}" for _ in range(c)) + "║\n").format(
            *[lig_names[ligne]] + [str(a) for a in tab[ligne]]) for ligne in range(len(tab))) + "╚" + "═" * (
                      l_first_col + 2) + ("╧══" + "═" * l_col) * c + "╝"
    return res


