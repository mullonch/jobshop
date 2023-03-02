from typing import Set, Iterable, Any, Sequence, List, Dict, Union, Optional, Self
from functools import singledispatchmethod
import numpy as np
import itertools



class Edge:
    """
    A simple class for edges of oriented graph.
    """
    def __init__(self, node_from:int|str, node_to:int|str, cost:float=0)->None:
        """
        Args:
            node_from (int|str): Origin point of the edge
            node_to (int|str): Destination of the edge
            cost (float, optional): Cost of the edge. Defaults to 0.
        """
        self.node_from:str = str(node_from)
        self.node_to:str = str(node_to)
        self.cost:float = cost

    def __str__(self)->str:
        """
        Returns:
            str: description of edge
        """
        return f"{self.node_from} -> {self.node_to} (cost = {self.cost})"
    
    def __iter__(self)->Iterable:
        return iter((self.node_from, self.node_to, self.cost))
    
    def __contains__(self, item:int|str)->bool:
        """Checks if given vertice is concerned by this edge

        Args:
            item (int|str): item to check

        Returns:
            bool: True if vertice is concerned by edge
        """
        return item in (self.node_from, self.node_to)
    
    def __eq__(self, item):
        return self.node_from == item.node_from and self.node_to == item.node_to and self.cost == item.cost
    
    def __hash__(self):
        return hash(tuple([self.node_from, self.node_to, self.cost]))

class Graph:
    """
        Simple base class for oriented graph
    """
    def __init__(self, *args)->None:
        """
        Constructor for the graph

        Args:
            * : anything, graph will be made with that
        """
        self.v:Set[str] = set()
        self.e:Set[Edge] = set()
        for elem in args:
            self.__iadd__(elem)

    @property
    def vertices(self)->Set[str]:
        """
        Renaming the V set in full name

        Returns:
            Set[Vertex]: the set of the vertices of the graph
        """
        return self.v

    @vertices.setter
    def set_vertices(self, vertices:Iterable[int|str]):
        """
        A Setter of the vertices of the graph (old vertices will be erased)

        Args:
            vertices (Iterable[Vertex]): new vertices of the graph
        """
        self.v = set((str(a) for a in vertices))
        self.e = set([edge for edge in self.e if edge.node_from in vertices and edge.node_to in vertices])

    @property
    def edges(self)->Set[Edge]:
        """
        Renaming the E set in full name

        Returns:
            Set[Edge]: the set of the edges of the graph
        """
        return self.e

    @edges.setter
    def set_edges(self, edges:Iterable[Edge]):
        """
        A Setter of the edges of the graph (old edges will be erased)

        Args:
            edges (Iterable[Edge]): new edges of the graph
        """
        self.e = set(edges)

    def add_vertex(self, vertex:int|str)->Self:
        """Add a new vertex to the graph.

        Args:
            vertex (str|int): new vertex

        Returns:
            Graph: self
        """
        self.v.add(str(vertex))
        return self

    def add_vertices(self, vertices:Iterable[int|str])->Self:
        """Add a new vertices to the graph

        Args:
            vertices (Iterable[str|int]): set of new vertices

        Returns:
            Graph: self
        """
        self.v.update(str(a) for a in vertices)
        return self

    def remove_vertex(self, vertex:int|str)->Self:
        """Remove a given vertex from the graph

        Args:
            vertex (str|int): vertex to remove

        Returns:
            Graph: self
        """
        self.e -= set([e for e in self.e if str(vertex) in e])
        self.v.remove(str(vertex))
        return self

    def add_edge(self, edge:Edge)->Self:
        """Add a new edge to the graph

        Args:
            edge (Edge): edge to add

        Returns:
            Graph: self
        """
        self.v.add(edge.node_from)
        self.v.add(edge.node_to)
        self.edges.add(edge)
        return self

    def link(self, v1:int|str, v2:int|str, cost:float=0)->Self:
        """Creates a new edge linking two given vertices at given cost

        Args:
            v1 (str|int): origin vertex
            v2 (str|int): destination vertex
            cost (float, optional): Cost to pass through this edge. Defaults to 0.

        Returns:
            Graph: self
        """
        self.add_vertices((v1, v2))
        self.add_edge(Edge(v1, v2, cost))
        return self

    def unlink(self, *vertices:int|str)->Self:
        """remove all edges linking the two given vertices

        Args:
            v1 (str|int): origin vertex
            v2 (str|int): destination vertex

        Returns:
            Graph: self
        """
        for i, v in enumerate(vertices):
            self.e -= set([e for e in self.e if e.node_from==vertices[i-1] and e.node_to==v])
        return self

    def __str__(self)->str:
        """
        Returns:
            str: quick description of graph
        """
        res = "V : " + str(self.v) + "\nE : "
        for arc in self.e:
            res += str(arc) + "\n"
        return res
    
    def __len__(self)->int:
        """
        Returns:
            int: number of vertices of the graph
        """
        return len(self.v)
    
    @property
    def order(self)->int:
        """
        Returns:
            int: number of vertices of the graph
        """
        return len(self.v)
    
    def __eq__(self, g:Self)->bool:
        """
        Args:
            g (Graph): graph to compare

        Returns:
            bool: True if the two graphs have the same set of vertices and edges
        """
        return self.v == g.v and self.e == g.e
    
    def __gt__(self, g:Self)->bool:
        return self.v > g.v and self.e > g.e
    
    def __ge__(self, g:Self)->bool:
        return self.v >= g.v and self.e >= g.e
    
    def __lt__(self, g:Self)->bool:
        return self.v < g.v and self.e < g.e
    
    def __le__(self, g:Self)->bool:
        return self.v <= g.v and self.e <= g.e

    @singledispatchmethod
    def __iadd__(self, item:Any)->Self:
        if isinstance(item, Graph):
            self.e.update(item.e)
            self.v.update(item.v)
        else:
            raise ValueError(f"element of type {type(item)} can't be interpreted as graph component.")
        return self
    
    @__iadd__.register
    def _(self, item:int)->Self:
        self.v.add(str(item))
        return self

    @__iadd__.register
    def _(self, item:str)->Self:
        self.v.add(item)
        return self

    @__iadd__.register
    def _(self, item:Edge)->Self:
        self.e.add(item)
        return self

    @__iadd__.register
    def _(self, item:list|tuple|set)->Self:
        for elem in item:
            self+=elem
        return self

    def __add__(self, item)->Self:
        res = self.clone
        res += item
        return res

    @singledispatchmethod
    def __isub__(self, item):
        if isinstance(item, Graph):
            self.e -= item.e
            self.v -= item.v
        else:
            raise ValueError(f"element of type {type(item)} can't be interpreted as graph component.")
    
    @__isub__.register
    def _(self, item:int):
        self.v.remove(str(item))

    @__isub__.register
    def _(self, item:str):
        self.v.remove(item)

    @__isub__.register
    def _(self, item:Edge):
        self.e.remove(item)

    @__isub__.register
    def _(self, item:list|tuple|set):
        for elem in item:
            self-=elem

    def __sub__(self, item)->Self:
        res = self.clone()
        res += item
        return res

    @singledispatchmethod
    def __contains__(self, item:Any)->bool:
        if isinstance(item, Graph):
            return all(v in self.v for v in item.v) and all(e in self.e for e in item.e)
        raise ValueError(f"element of type {type(item)} can't be interpreted as graph component.")
    
    @__contains__.register
    def _(self, item:int)->bool:
        return str(item) in self.v

    @__contains__.register
    def _(self, item:str)->bool:
        return item in self.v

    @__contains__.register
    def _(self, item:Edge)->bool:
        return item in self.e

    @__contains__.register
    def _(self, item:list|tuple|set)->bool:
        return all(elem in self for elem in item)
    
    def clone(self)->Self:
        return Graph(self.v, self.e)
    
    def clear(self)->Self:
        self.v = set()
        self.e = set()
        return self

    def clear_edges(self)->Self:
        self.e = set()
        return self

    def __bool__(self)->bool:
        return not len(self.e)
    
    def __hash__(self):
        return hash(tuple(self.e) + tuple(self.v))
    
    def __and__(self, g:Self)->Self:
        return Graph(self.e & g.e, self.v & g.v)
    
    def __or__(self, g:Self)->Self:
        return Graph(self.e | g.e, self.v | g.v)
    
    def __xor__(self, g:Self)->Self:
        return Graph(self.e ^ g.e, self.v ^ g.v)
    
    def __iand__(self, g:Self)->None:
        self.v &= g.v
        self.e &= g.e

    def __ior__(self, g:Self)->None:
        self.v |= g.v
        self.e |= g.e

    def __ixor__(self, g:Self)->None:
        self.v ^= g.v
        self.e ^= g.e

    def link_all(self, node_list=None, cost=0):
        if node_list is None:
            node_list = self.v
        for n1, n2 in itertools.combinations(node_list, 2):
            self.link(n1, n2, cost)

    def adjacence_matrix(self)->np.ndarray:
        node_list = list(self.v)
        res = np.zeros((len(node_list), len(node_list)))
        for e in self.e:
            res[node_list.index(e.node_from)][node_list.index(e.node_to)] = 1
        return res
    
    def degree(self, vertex:Union[str, int])->int:
        return len([edge for edge in self.e if edge.node_from == vertex])
    
    def outgoing_edges(self, vertex:Union[str, int])->Set[Edge]:
        return set([edge for edge in self.e if edge.node_from == vertex])

    def incoming_edges(self, vertex:Union[str, int])->Set[Edge]:
        return set([edge for edge in self.e if edge.node_to == vertex])
    
    def neighbors(self, vertex:Union[str, int])->Set[str]:
        return set([arc.node_to for arc in self.e if arc.node_from == str(vertex)])

    def incomings(self, vertex:Union[str, int])->Set[str]:
        return set([arc.node_from for arc in self.e if arc.node_to == str(vertex)])
    
    def accessible_from(self, v_start:Union[str, int]):
        v_past = set((str(v_start),))
        v_next = self.neighbors(v_start)
        while v_next!=set():
            vertice = v_next.pop()
            v_past.add(vertice)
            v_next.update(self.neighbors(vertice))
            v_next = v_next - v_past
        return v_past

    def are_linked(self, node_from:Union[str, int], node_to:Union[str, int])->bool:
        return self.get_edge(str(node_from), str(node_to)) is not None

    def get_edge(self, node_from, node_to)->Optional[Edge]:
        for a in self.e:
            if a.node_from == str(node_from) and a.node_to == str(node_to):
                return a
        return None

    def get_cost(self, node_from:Union[str, int], node_to:Union[str, int]):
        edge = self.get_edge(node_from, node_to)
        if edge:
            return edge.cost
        return float("inf")
    
    def shorter_path(self, node_from:Union[str, int], node_to:Union[str, int])->Union[List[str], int]:
        # Algo de Dijkstra
        p = set()
        d = dict.fromkeys(self.v, float("inf"))
        d[str(node_from)] = 0
        predecesseurs = dict()
        while (self.v - p) != set():
            a = next(iter(self.v - p))
            for elem in self.v - p:
                if d[elem] < d[a]:
                    a = elem
            p.add(a)
            for b in self.neighbors(a):
                if d[b] > d[a] + self.get_cost(a, b):
                    d[b] = d[a] + self.get_cost(a, b)
                    predecesseurs[b] = a
        path = [str(node_to)]
        while path[0] != str(node_from):
            if path[0] not in predecesseurs.keys():
                return -1
            path = [predecesseurs[path[0]]] + path
        return path

    def path_length(self, path:Union[List[str],int])->float:
        if type(path) is list:
            length = 0
            for i in range(1, len(path)):
                length += self.get_cost(path[i - 1], path[i])
            return length
        return float("inf")

    def shorter_path_length(self, node_from:Union[str, int], node_to:Union[str, int])->float:
        return self.path_length(self.shorter_path(str(node_from), str(node_to)))

    def topological_list(self):
        explored = []
        start_node = next(node for node in self.v if len(self.incomings(node)) == 0)
        stack = [start_node]
        while len(stack) > 0:
            to_explore = list(set(self.neighbors(stack[-1])) - set(explored))
            if len(to_explore) == 0:
                explored += [stack.pop(-1)]
            stack += to_explore
        explored.reverse()
        return explored

    def is_path(self, node_from:Union[str, int], node_to:Union[str, int], level:int=0)->bool:
        if(level > 50):
            raise RuntimeError("A lot of recursions have been made in depth first search")
        # Parcours en profondeur (récursif)
        if node_from == node_to and level != 0:
            return True
        return any([self.is_path(next_node, node_to, level + 1) for next_node in self.neighbors(node_from)])

    def is_cycle_from_node(self, node:Union[str, int])->bool:
        explored = []
        stack = [node]
        while len(stack) > 0:
            if node in self.neighbors(stack[-1]):
                return True
            to_explore = list(set(self.neighbors(stack[-1])) - set(explored))
            if len(to_explore) == 0:
                explored += [stack.pop(-1)]
            stack += to_explore
        return False

    def has_cycle(self)->bool:
        return any(self.is_cycle_from_node(n) for n in self.v)
