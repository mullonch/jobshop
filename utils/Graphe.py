import itertools



class Graphe:
    def __init__(self, construction=[]):
        self.V = set()  # Noeuds
        self.E = set()  # Arretes
        for elem in construction:
            self.__add__(elem)
        # self.oriented=True

    def __str__(self):
        res = "V : " + str(self.V) + "\nE : "
        for arc in self.E:
            res += str(arc) + "\n"
        return res

    def __len__(self):
        return len(self.V)

    def __contains__(self, item):
        if isinstance(item, Edge):
            return item in self.E
        if isinstance(item, Graphe):
            return all(v in self.V for v in item.V) and all(e in self.E for e in item.E)
        return str(item) in self.V

    def __eq__(self, g):
        if isinstance(g, Graphe):
            return self.V == g.V and self.E == g.E
        return False

    def __hash__(self):
        return hash(tuple(self.E) + tuple(self.V))

    def __ne__(self, g):
        if isinstance(g, Graphe):
            return self.V != g.V or self.E != g.E
        return False

    def __add__(self, e):
        if isinstance(e, str):
            self.V.add(e)
        elif isinstance(e, list) or isinstance(e, tuple):
            for elem in e:
                self.__add__(elem)
        elif isinstance(e, dict):
            for key in e:
                self.link((key[0], key[1]), e[key])
        elif isinstance(e, Edge):
            self.E.add(e)
        elif isinstance(e, Graphe):
            for v in e.V:
                self.V.add(v)
            for e in e.E:
                self.E.add(e)
        else:
            raise Exception("Cannot add " + str(type(e)) + "to Graphe.")
        return self

    def __sub__(self, e):
        if isinstance(e, str):
            self.V.remove(e)
        elif isinstance(e, Edge):
            self.E.remove(e)
        elif isinstance(e, Graphe):
            for v in e.V:
                self.V.remove(v)
            for e in e.E:
                self.E.remove(e)
        else:
            raise "Cannot remove " + str(type(e)) + "to Graphe."
        return self

    def adjacence_matrix(self):
        node_list = list(self.V)
        res = [[0 for _ in self.V] for _ in self.V]
        for a in self.E:
            res[node_list.index(a.node_from)][node_list.index(a.node_to)] = 1
        return res

    def add(self, elem):
        self.__add__(elem)


    def remove_edge(self, node_from, node_to):
        to_remove = [edge for edge in self.E if edge.node_from == node_from and edge.node_to == node_to]
        for edge in to_remove:
            self.E.remove(edge)


    def unlink(self, node_list):
        for i in range(1, len(node_list)):
            self.remove_edge(node_list[i-1], node_list[i])

    def link(self, node_list, cost=0, oriented=True):
        self.__add__(node_list)
        for i in range(1, len(node_list)):
            self.E.add(Edge(node_list[i - 1], node_list[i], cost))
            if not oriented:
                self.E.add(Edge(node_list[i], node_list[i - 1], cost))

    def link_all(self, node_list=None, cost=0):
        if node_list is None:
            node_list = self.V
        for n1, n2 in itertools.combinations(node_list, 2):
            self.link((n1, n2), cost, oriented=False)

    def get_neighbors(self, node):
        return [arc.node_to for arc in self.E if arc.node_from == str(node)]

    def get_incomings(self, node):
        return [arc.node_from for arc in self.E if arc.node_to == str(node)]

    def are_linked(self, node_from, node_to):
        return self.get_arrete(str(node_from), str(node_to)) is not None

    def get_arrete(self, node_from, node_to):
        for a in self.E:
            if a.node_from == str(node_from) and a.node_to == str(node_to):
                return a
        return None

    def get_cost(self, node_from, node_to):
        if self.are_linked(str(node_from), str(node_to)):
            return self.get_arrete(str(node_from), str(node_to)).cost
        return float("inf")

    def shorter_path(self, node_from, node_to):
        # Algo de Dijkstra
        p = set()
        d = dict.fromkeys(self.V, float("inf"))
        d[str(node_from)] = 0
        predecesseurs = dict()
        while (self.V - p) != set():
            a = next(iter(self.V - p))
            for elem in self.V - p:
                if d[elem] < d[a]:
                    a = elem
            p.add(a)
            for b in self.get_neighbors(a):
                if d[b] > d[a] + self.get_cost(a, b):
                    d[b] = d[a] + self.get_cost(a, b)
                    predecesseurs[b] = a
        path = [str(node_to)]
        while path[0] != str(node_from):
            if path[0] not in predecesseurs.keys():
                return -1
            path = [predecesseurs[path[0]]] + path
        return path

    def path_length(self, path):
        if path == -1:
            return float("inf")
        length = 0
        for i in range(1, len(path)):
            length += self.get_cost(path[i - 1], path[i])
        return length

    def shorter_path_length(self, node_from, node_to):
        return self.path_length(self.shorter_path(str(node_from), str(node_to)))

    def has_node(self, node):
        return str(node) in self.V

    def topological_list(self):
        res = []
        s = set([node for node in self.V if len(self.get_incomings(node)) == 0])
        while len(s) > 0:
            n = s.pop()
            if n in res:
                res.remove(n)
            res.append(n)
            for m in self.get_neighbors(n):
                s.add(m)
        return res

    def is_path(self, node_from, node_to, level=0):
        return self.shorter_path(node_from, node_to)!= -1
        # Parcours en profondeur (récursif)
        # if node_from == node_to and level != 0:
        #     return True
        # return any([self.is_path(next_node, node_to, level + 1) for next_node in self.get_neighbors(node_from)])
        

    def has_cycle(self):
        return any(self.is_path(n, n) for n in self.V)


class Edge:
    def __init__(self, node_from, node_to, cost=0):
        self.node_from = str(node_from)
        self.node_to = str(node_to)
        self.cost = cost

    #def __eq__(self, e):
        #if(isinstance(e, Edge)):
        #    return e.node_from == self.node_from and e.node_to == self.node_to and self.cost == e.cost
        #return False

    def __str__(self):
        return self.node_from + " -> " + self.node_to + " (cout = " + str(self.cost) + ")"
