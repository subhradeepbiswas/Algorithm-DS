# -*- coding: utf-8 -*-
"""
SER501 Assignment 4 scaffolding code
created by: Xiangyu Guo
#ToDo
author:
"""
import numpy as np

# ToDo

# ============================== Counting Pond ================================


def count_ponds(G):
    m = len(G)
    n = len(G[0])

    if(len(G) == 0):
        return 0

    total = 0

    visited = [[0 for j in range(len(G[0]))]for i in range(len(G))]

    for i in range(0, m):
        for j in range(0, n):

            if(visited[i][j] == 0 and G[i][j] == '#'):
                total = total + 1

                DFS(G, i, j, visited)
    return total


def DFS(G, i, j, visited):

    x = [-1, -1, -1, 0, 0, 1, 1, 1]
    y = [-1, 0, 1, -1, 1, -1, 0, 1]

    visited[i][j] = 1

    for d in range(0, 8):
        a = x[d] + i
        b = y[d] + j

        if(a >= 0 and b >= 0 and a < len(G) and b < len(G[0]) and
           visited[a][b] == 0 and G[a][b] == '#'):
                DFS(G, a, b, visited)


# ======================== Longest Ordered Subsequence ========================


def longest_ordered_subsequence(L):
    n = len(L)

    if(n == 0):
        return 0
    if(n == 1):
        return 1

    final = [1 for i in range(n)]

    for i in range(1, n):
        for j in range(0, i):
            if(L[i] > L[j]):
                if(final[j] + 1 > final[i]):
                    final[i] = final[j] + 1
    check = 0

    for i in range(0, len(final)):
        if(final[i] > final[check]):
            check = i
    return final[check]

# =============================== Supermarket =================================


def supermarket(Items):
    n = len(Items)

    if(n == 0):
        return 0

    items = sorted(Items, key=lambda x: x[1])

    maxT = items[-1][1]

    B = [0 for i in range(n)]

    for i in range(0, n):
        B[i] = items[i][0]

    for i in range(1, n):
        for j in range(0, i):

            if(items[j][1] != items[i][1]):
                if(B[j] + items[i][0] > B[i]):
                    B[i] = items[i][0] + B[j]

            if(items[j][1] == items[i][1]) and (maxT == items[j][1]):
                # if(B[j] + items[i][0] > B[i]):
                B[i] = items[i][0] + items[j][0]

    return np.amax(B)


# =============================== Unit tests ==================================


def test_suite():

    if count_ponds(["#--------##-",
                    "-###-----###",
                    "----##---##-",
                    "---------##-",
                    "---------#--",
                    "--#------#--",
                    "-#-#-----##-",
                    "#-#-#-----#-",
                    "-#-#------#-",
                    "--#-------#-"]) == 3:
        print('passed')
    else:
        print('failed')

    if longest_ordered_subsequence([1, 7, 3, 5, 9, 4, 8]) == 4:
        print('passed')
    else:
        print('failed')

    if supermarket([(50, 2), (10, 1), (20, 2), (30, 1)]) == 80:
        print('passed')
    else:
        print('failed')

    # ToDo More test cases


if __name__ == '__main__':
    test_suite()
