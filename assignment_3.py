# -*- coding: utf-8 -*-
"""
SER501 Assignment 3 scaffolding code
created by: Xiangyu Guo
"""
# from collections import deque
import sys
import math
# =============================================================================


def popmin(pqueue):

    lowest = math.inf
    keylowest = None
    for key in pqueue:
        if pqueue[key] < lowest:
            lowest = pqueue[key]
            keylowest = key
    del pqueue[keylowest]
    return keylowest


def dfs(graph, temp_vertices, node, discover, disTime, finTime):

    if node not in discover:
        discover.append(node)
        idx = temp_vertices.index(node)
        disTime[idx] = max(max(disTime), max(finTime))+1

        for n in graph[node]:
            dfs(graph, temp_vertices, n, discover, disTime, finTime)
        idx = temp_vertices.index(node)
        finTime[idx] = max(max(disTime), max(finTime))+1

    return [discover, disTime, finTime]


class Graph(object):
    """docstring for Graph"""
    user_defined_vertices = []
    dfs_timer = 0

    def __init__(self, vertices, edges):
        super(Graph, self).__init__()
        n = len(vertices)
        self.matrix = [[0 for x in range(n)] for y in range(n)]
        self.vertices = vertices
        self.edges = edges
        for edge in edges:
            x = vertices.index(edge[0])
            y = vertices.index(edge[1])
            self.matrix[x][y] = edge[2]

    def display(self):
        print(self.vertices)
        for i, v in enumerate(self.vertices):
            print(v, self.matrix[i])

    def transpose(self):
        # ToDo
        print("Transpose of the graph")
        edgesT = []
        for i in range(0, len(self.matrix)):
            for j in range(0, len(self.matrix[i])):
                if self.matrix[i][j] != 0:
                    edgesT.append((self.vertices[j],
                                   self.vertices[i], self.matrix[i][j]))

        graph = Graph(self.vertices, edgesT)

        # print(graphT)
        graph.display()
        # print(edgesT)

    def in_degree(self):
        print("In degree of the graph:")
        # ToDo
        count = [0]*len(self.vertices)
        for i in range(0, len(self.vertices)):
            for j in range(0, len(self.matrix[:][i])):
                if self.matrix[j][i] > 0:
                    count[i] += 1
        # print("Not implemented yet!")
        # ToDo: invoke print_degree to print out the final result.
        self.print_degree(count)

    def out_degree(self):
        print("Out degree of the graph:")
        count = [0] * len(self.vertices)
        for i in range(0, len(self.matrix)):
            for j in range(0, len(self.matrix[i])):
                if self.matrix[i][j] > 0:
                    count[i] += 1
        # print("Not implemented yet!")
        # ToDo: invoke print_degree to print out the final result.
        self.print_degree(count)

    def dfs_on_graph(self):
        # ToDo
        print("DFS on Graph implementation")
        graph_al = dict((k, []) for k in self.vertices)
        # graph_al = ('k': [] for k in self.vertices)
        for i in self.edges:
            if i[0] in graph_al:
                graph_al[i[0]].append(i[1])
            else:
                graph_al[i[0]] = [i[1]]

        # graph = Graph(self.vertices, graph_al)
        # print (self.vertices)

        temp_vertices = [i for i in self.vertices]
        # print(temp_vertices)

        minV = min(temp_vertices)
        # print("Min value: " +minV)

        path = [[], [0 for i in self.vertices], [0 for i in self.vertices]]

        while len(self.vertices) > len(path[0]):

            if len(temp_vertices) > 0:
                minV = min(temp_vertices)
                # print("Min V:     " + minV)
            path = dfs(graph_al, self.vertices, minV,
                       path[0], path[1], path[2])

            for i in path[0]:
                if i in temp_vertices:
                    del temp_vertices[temp_vertices.index(i)]

        self.print_discover_and_finish_time(path[1], path[2])

    def prim(self, root):
        # ToDo
        print("Prim's algorithm implementation")
        parent = {}
        key = {}
        pqueue = {}

        graph_al = dict((k, {}) for k in self.vertices)
        # graph_al = ('k': [] for k in self.vertices)
        for i in self.edges:
            if i[0] in graph_al:
                graph_al[i[0]].update({i[1]: i[2]})
            else:
                graph_al[i[0]] = {i[1]: i[2]}

        for v in graph_al:
            parent[v] = None
            key[v] = math.inf

        key[root] = 0
        parent[root] = None

        for v in graph_al:
            pqueue[v] = key[v]

        i = 1
        # d = [k for k in graph_al]
        keyList = [sys.maxsize for i in self.vertices]
        piList = ['None' for i in self.vertices]

        while pqueue:
            u = popmin(pqueue)
            for v in graph_al[u]:
                if ((v in pqueue) & (graph_al[u][v] < key[v])):
                    parent[v] = u
                    key[v] = graph_al[u][v]
                    pqueue[v] = graph_al[u][v]

            for v in key:
                vi = self.vertices.index(v)
                keyList[vi] = key[v]

            for v in parent:
                vi = self.vertices.index(v)
                piList[vi] = parent[v]

            self.print_d_and_pi(i, keyList, piList)
            i += 1

        return parent

    def bellman_ford(self, root):
        # ToDo
        print("Bellman_Ford implementation")
        edges = [None] * len(self.vertices)
        rootIdx = self.vertices.index(root)
        dist = [None] * len(self.vertices)
        parent = [0] * len(self.vertices)
        parentinit = ['None'] * len(self.vertices)
        parent1 = ['a'] * len(self.vertices)

        for i in range(len(self.vertices)):
            if(i == rootIdx):
                dist[i] = 0
            else:
                dist[i] = math.inf
        self.print_d_and_pi(-1, dist, parentinit)

        for p in range(len(self.vertices)):
            idx = list(self.vertices).index(self.vertices[p])
            edges[p] = idx

        for i in range(len(self.vertices) - 1):
            for u, v, w in self.edges:

                idxu = list(self.vertices).index(u)
                idxv = list(self.vertices).index(v)
                rootIdx = list(self.vertices).index(root)
                if ((dist[idxu] != float("Inf")) and
                        (dist[idxu] + w < dist[idxv])):
                    dist[idxv] = dist[idxu] + w
                    parent[idxv] = idxu
                    parent[rootIdx] = 'None'
                elif ((dist[idxu] == float("Inf")) and
                        (dist[idxv] == float("Inf"))):
                    parent[idxv] = 'None'

            for n in range(len(self.vertices)):
                if(parent[n] != 'None'):
                    parent1[n] = self.vertices[parent[n]]
                else:
                    parent1[n] = 'None'
            self.print_d_and_pi(i, dist, parent1)

        for u, v, w in self.edges:
            idxu = self.vertices.index(u)
            idxv = self.vertices.index(v)
            if ((dist[idxu] != float("Inf")) and
                    (dist[idxu] + w < dist[idxv])):
                print("Graph contains negative cycle")
                return

    def dijkstra(self, root):
        # ToDo
        print("Dijkstra implementation")
        parent = {}  # pair {vertex: predecesor in MST}
        key = {}  # keep track of minimum weight for each vertex
        pqueue = {}  # priority queue implemented as list

        graph_al = dict((k, {}) for k in self.vertices)
        # graph_al = ('k': [] for k in self.vertices)
        for i in self.edges:
            if i[0] in graph_al:
                graph_al[i[0]].update({i[1]: i[2]})

            else:
                graph_al[i[0]] = {i[1]: i[2]}

        for v in graph_al:
            parent[v] = -1
            key[v] = sys.maxsize

        key[root] = 0
        parent[root] = None

        for v in graph_al:
            pqueue[v] = key[v]

        i = 0
        keyList = [sys.maxsize for i in self.vertices]
        rootindex = self.vertices.index(root)
        keyList[rootindex] = 0
        piList = ['None' for i in self.vertices]
        self.print_d_and_pi(i, keyList, piList)
        i = 1
        while pqueue:
            u = popmin(pqueue)
            # print(u)
            for v in graph_al[u]:
                if ((v in pqueue) & ((graph_al[u][v]+key[u]) < key[v])):
                    parent[v] = u
                    # print("parent pf v" , parent[v])
                    key[v] = (graph_al[u][v] + key[u])
                    pqueue[v] = (graph_al[u][v] + key[u])

            for v in key:
                vi = self.vertices.index(v)
                keyList[vi] = key[v]
                if(parent[v] == -1):
                    parent[v] = 'None'
                piList[vi] = parent[v]
                # print(keyList[vi])

            # for v in parent:
                # vi = self.vertices.index(v)
                # print(vi)
                # print(parent[v])
                # if(parent[v] == -1):
                # parent[v] = 'None'
                # piList[vi] = parent[v]

            self.print_d_and_pi(i, keyList, piList)

            i += 1

    def print_d_and_pi(self, iteration, d, pi):
        assert((len(d) == len(self.vertices)) and
               (len(pi) == len(self.vertices)))
        print("Iteration: {0}".format(iteration))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\td: {1}\tpi: {2}".format(v, 'inf' if d[i]
                                                        == sys.maxsize
                                                        else d[i], pi[i]))

    def print_discover_and_finish_time(self, discover, finish):
        assert((len(discover) == len(self.vertices)) and
               (len(finish) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDiscovered: {1}\tFinished: {2}".format(
                    v, discover[i], finish[i]))

    def print_degree(self, degree):
        assert((len(degree) == len(self.vertices)))
        for i, v in enumerate(self.vertices):
            print("Vertex: {0}\tDegree: {1}".format(v, degree[i]))
            
def test_transpose():
    graph = Graph(['1', '2', '3', '4', '5', '6'],
                  [('1', '2', 6),
                   ('2', '5', 3),
                   ('5', '6', 6),
                   ('6', '4', 2),
                   ('4', '1', 5),
                   ('3', '1', 1),
                   ('3', '2', 5),
                   ('3', '5', 6),
                   ('3', '6', 4),
                   ('3', '4', 5)])
    # transpose 2pts
    graph.transpose()
    graph.display()

def test_dfs():
    # Q3 5 pts dfs output
    # 5 pts write-up
    graph = Graph(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                  [('a', 'b', 1),
                   ('a', 'f', 1),
                   ('b', 'c', 1),
                   ('b', 'i', 1),
                   ('b', 'g', 1),
                   ('c', 'd', 1),
                   ('c', 'i', 1),
                   ('d', 'e', 1),
                   ('d', 'g', 1),
                   ('d', 'h', 1),
                   ('d', 'i', 1),
                   ('e', 'f', 1),
                   ('e', 'h', 1),
                   ('f', 'g', 1),
                   ('g', 'h', 1)])
    try:
        graph.dfs_on_graph()
    except:
        print("Failed on DFS")

def test_prim():
    # Q4 - Prim 10 pts (One set, each iteration)
    graph = Graph(['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'J'],
                  [('A', 'B', 5),
                   ('A', 'G', 22),
                   ('A', 'E', 7),
                   ('B', 'G', 1),
                   ('E', 'G', 4),
                   ('B', 'J', 10),
                   ('G', 'J', 8),
                   ('J', 'I', 20),
                   ('G', 'I', 21),
                   ('G', 'C', 2),
                   ('E', 'C', 15),
                   ('E', 'D', 26),
                   ('D', 'C', 9),
                   ('C', 'I', 6),
                   ('D', 'H', 8),
                   ('C', 'H', 3),
                   ('H', 'I', 11),
                   ('B', 'A', 5),
                   ('G', 'A', 22),
                   ('E', 'A', 7),
                   ('G', 'B', 1),
                   ('G', 'E', 4),
                   ('J', 'B', 10),
                   ('J', 'G', 8),
                   ('I', 'J', 20),
                   ('I', 'G', 21),
                   ('C', 'G', 2),
                   ('C', 'E', 15),
                   ('D', 'E', 26),
                   ('C', 'D', 9),
                   ('I', 'C', 6),
                   ('H', 'D', 8),
                   ('H', 'C', 3),
                   ('I', 'H', 11)])
    try:
        graph.prim('G')
    except:
        print("Failed on Prim")


def test_bellman_ford():
    # Q5 - Bellman Ford 10 pts (2 set of test cases)
    graph = Graph(['A', 'B', 'C', 'D', 'E'],
                  [('A', 'B', -1),
                   ('A', 'C', 4),
                   ('B', 'C', 3),
                   ('B', 'D', 2),
                   ('B', 'E', 2),
                   ('D', 'B', 1),
                   ('D', 'C', 5),
                   ('E', 'D', -3)])
    try:
        graph.bellman_ford('A')
    except:
        print("Failed on Bellman-ford")


def test_bellman_ford_alt():
    # Q5 alternate
    graph = Graph(['A', 'B', 'C', 'D', 'E'],
                  [('A', 'B', -1),
                   ('A', 'C', 4),
                   ('B', 'C', 3),
                   ('B', 'D', 2),
                   ('B', 'E', 2),
                   ('D', 'B', -1),
                   ('D', 'C', 5),
                   ('E', 'D', -3)])
    try:
        graph.bellman_ford('A')
    except:
        print("Failed on Bellman-ford-alt")


def test_dijkstra():
    # Q6 - Dijkstra 10 pts (10 pts for 10 iterations )
    graph = Graph(['1', '2', '3', '4', '5', '6'],
                  [('1', '2', 6),
                   ('2', '5', 3),
                   ('5', '6', 6),
                   ('6', '4', 2),
                   ('4', '1', 5),
                   ('3', '1', 1),
                   ('3', '2', 5),
                   ('3', '5', 6),
                   ('3', '6', 4),
                   ('3', '4', 5)])
    try:
        graph.dijkstra('3')
    except:
        print("Failed on Dijkstra")


def main():
    
    test_prim()
    # Thoroughly test your program and produce useful output.
    # Q1 and Q2
#     graph = Graph(['1', '2'], [('1', '2', 1)])
#     graph.display()
#     graph.transpose()
#     graph.display()
#     graph.transpose()
#     graph.display()
#     graph.in_degree()
#     graph.out_degree()
#     graph.print_d_and_pi(1, [1, sys.maxsize], [2, None])
#     graph.print_degree([1, 0])
#     graph.print_discover_and_finish_time([0, 2], [1, 3])

#     # Q3
#     graph = Graph(['q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
#                   [('q', 's', 1),
#                    ('s', 'v', 1),
#                    ('v', 'w', 1),
#                    ('w', 's', 1),
#                    ('q', 'w', 1),
#                    ('q', 't', 1),
#                    ('t', 'x', 1),
#                    ('x', 'z', 1),
#                    ('z', 'x', 1),
#                    ('t', 'y', 1),
#                    ('y', 'q', 1),
#                    ('r', 'y', 1),
#                    ('r', 'u', 1),
#                    ('u', 'y', 1)])
#     graph.display()
#     graph.dfs_on_graph()

#     # Q4 - Prim
#     graph = Graph(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
#                   [('A', 'H', 6),
#                    ('H', 'A', 6),
#                    ('A', 'B', 4),
#                    ('B', 'A', 4),
#                    ('B', 'H', 5),
#                    ('H', 'B', 5),
#                    ('B', 'C', 9),
#                    ('C', 'B', 9),
#                    ('G', 'H', 14),
#                    ('H', 'G', 14),
#                    ('F', 'H', 10),
#                    ('H', 'F', 10),
#                    ('B', 'E', 2),
#                    ('E', 'B', 2),
#                    ('G', 'F', 3),
#                    ('F', 'G', 3),
#                    ('E', 'F', 8),
#                    ('F', 'E', 8),
#                    ('D', 'E', 15),
#                    ('E', 'D', 15)])
#     graph.prim('G')

#     # Q5
#     graph = Graph(['s', 't', 'x', 'y', 'z'],
#                   [('t', 'x', 5),
#                    ('t', 'y', 8),
#                    ('t', 'z', -4),
#                    ('x', 't', -2),
#                    ('y', 'x', -3),
#                    ('y', 'z', 9),
#                    ('z', 'x', 7),
#                    ('z', 's', 2),
#                    ('s', 't', 6),
#                    ('s', 'y', 7)])
#     graph.bellman_ford('z')

#     # Q5 alternate
#     graph = Graph(['s', 't', 'x', 'y', 'z'],
#                   [('t', 'x', 5),
#                    ('t', 'y', 8),
#                    ('t', 'z', -4),
#                    ('x', 't', -2),
#                    ('y', 'x', -3),
#                    ('y', 'z', 9),
#                    ('z', 'x', 4),
#                    ('z', 's', 2),
#                    ('s', 't', 6),
#                    ('s', 'y', 7)])
#     graph.bellman_ford('s')

#     # Q6
#     graph = Graph(['s', 't', 'x', 'y', 'z'],
#                   [('s', 't', 3),
#                    ('s', 'y', 5),
#                    ('t', 'x', 6),
#                    ('t', 'y', 2),
#                    ('x', 'z', 2),
#                    ('y', 't', 1),
#                    ('y', 'x', 4),
#                    ('y', 'z', 6),
#                    ('z', 's', 3),
#                    ('z', 'x', 7)])
#     graph.dijkstra('s')


if __name__ == '__main__':
    main()
