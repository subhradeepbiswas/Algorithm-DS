# -*- coding: utf-8 -*-

from numpy import asarray
import math

ENERGY_LEVEL = [100, 113, 110, 85, 105, 102, 86, 63, 81, 101, 94, 106, 101, 79, 94, 90, 97]

# ==============================================================


# The brute force method to solve first problem
def find_significant_energy_increase_brute(A):

    """
    Return a tuple (i,j) where A[i:j] is the most
    significant energy increase period.
    time complexity = O(n^2)
    """

    alen = len(A)
    diff = -1
    max_index = 0
    # Loop i starts from the end of the loop
    for i in range(alen - 1, 0, -1):

        # if there is an element thats higher than the last elem,
        # max_index is updated with new highest value index
        if A[i] > A[max_index]:
            max_index = i

        # Loop j runs through the elemts before i
        # if it finds a smaller elem that creates more than the
        # last captured difference from max index, then updates result
        for j in range(0, (i + 1)):
            if ((A[i] - A[j]) > diff):
                diff = A[i] - A[j]
                result = [j, i]

    # Base case: if max energy at the 1st element
    if max_index == 0:
        result = [0, 1]

    return result[0], result[1]


# ==============================================================

def crossing_diff(arr, l, m, h):

    diff = 0
    left_diff = 0
    min_idx = m

    # Approaching left to include lower energy level
    # Min index updated accordingly
    for i in range(m, (l - 1), -1):
        diff = (arr[m] - arr[i])
        if (diff > left_diff):
            left_diff = diff
            min_idx = i

    max_idx = m
    diff = 0
    right_diff = 0

    # Approaching right to include higher energy level
    # Max index updated accordingly
    for i in range(m, h):
        diff = (arr[i] - arr[m])
        if (right_diff < diff):
            right_diff = diff
            max_idx = i

    result = [min_idx, max_idx]
    return result


# The recursive method to solve first problem
def find_significant_energy_increase_recursive(A):

    """
    Return a tuple (i,j) where A[i:j] is the most significant
    energy increase period.
    time complexity = O (n logn)
    """
    # Expecting -100 as last element for internal fn call
    # Later explained in detail
    if A[len(A) - 1] == -100:
        A.pop()
    else:
        if A[0] == max(A):
            return 0, 1

    if (len(A) == 1):
        return 0, 0

    m = len(A) // 2

    # Assigning 1st half of array
    # appending -100 as last elem so that it denotes
    # internal function call
    subarr1 = A[0:m]
    subarr1.append(-100)
    result_left = find_significant_energy_increase_recursive(subarr1)

    # Assigning 1st half of array
    # appending -100 as last elem so that it denotes
    # internal function call
    subarr2 = A[m:]
    subarr2.append(-100)
    result_right = find_significant_energy_increase_recursive(subarr2)

    # Overlapping array across mid
    result_mid = crossing_diff(A, 0, m, len(A))

    # Finding Max difference among all intermediate o/p,
    # it makes our result
    max_diff = max((A[result_mid[1]] - A[result_mid[0]]),
                   (A[m + result_right[1]] - A[m + result_right[0]]),
                   (A[result_left[1]] - A[result_left[0]]))
    if ((A[result_mid[1]] - A[result_mid[0]]) == max_diff):
        result = [result_mid[0], result_mid[1]]
    if ((A[m + result_right[1]]-A[m + result_right[0]]) == max_diff):
        result = [m + result_right[0], m + result_right[1]]
    if ((A[result_left[1]] - A[result_left[0]]) == max_diff):
        result = [result_left[0], result_left[1]]

    return result[0], result[1]


# ==============================================================

# The iterative method to solve first problem
def find_significant_energy_increase_iterative(A):

    """
    Return a tuple (i,j) where A[i:j] is the most
    significant energy increase period.
    time complexity = O(n)
    """

    # Only loop to generate O(n) complexity
    for i in range(0, len(A)):
        if (i == 0):
            min_idx = i
            max_idx = i
            future_min_idx = i

        else:
            # if a min elem found, checks if it appears b4 max index
            # if its before max idx, update min_index
            # else, Stores it for future if a bigger elem found later
            if (A[i] < A[min_idx]):
                if (i > max_idx):
                    if (A[i] < A[future_min_idx]):
                        future_min_idx = i
                else:
                    min_idx = i

            # if a bigger elem found than last max elem, max index updated
            # if future_index appears before new max,
            # min index updated with future idx
            if (A[i] > A[max_idx]):
                max_idx = i
                if max_idx > future_min_idx:
                    min_idx = future_min_idx

            # even if curr elem is not bigger than max,
            # it compares diff, min max updated accordingly
            else:
                if (max_idx != 0):
                    if (
                        (A[i] - A[future_min_idx]) > (A[max_idx] - A[min_idx])):
                        min_idx = future_min_idx
                        max_idx = i

    # Base case: array begins with max elem
    if (max_idx == 0):
        result = [0, 1]
    else:
        result = [min_idx, max_idx]

    return result[0], result[1]


# ==============================================================
# Two functions added for Strassens' algorithm implementation
# matAdd function adds two matrixes
def matAdd(A, B):
    n = len(A)
    C = [[0 for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j] + B[i][j]
    return C


# matSub function subtracts two matrixes
def matSub(A, B):
    n = len(A)
    C = [[0 for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            C[i][j] = A[i][j] - B[i][j]
    return C


# ==============================================================
# The Strassen Algorithm to do the matrix multiplication
def square_matrix_multiply_strassens(A, B):

    """
    Return the product AB of matrix multiplication.
    Assume len(A) is a power of 2
    """

    A = asarray(A)

    B = asarray(B)

    assert A.shape == B.shape

    assert A.shape == A.T.shape

    assert (len(A) & (len(A) - 1)) == 0, "A is not a power of 2"

    # Recurse until matrix size 2X2
    if len(A) > 2:
        # submat1=A[0:math.floor(len(A))]
        # submat2=A[math.floor(len(A)):]
        matHalfA = int(len(A) / 2)
        matHalfB = int(len(B) / 2)

        # A00=[[0]*matHalfB]*matHalfA
        # A01=[[0]*matHalfB]*matHalfA
        # A10=[[0]*matHalfB]*matHalfA
        # A11=[[0]*matHalfB]*matHalfA

        # Creating new list[list] to hold divided parts of main list A
        A00 = [[0 for j in range(
            0, matHalfB)] for i in range(0, matHalfA)]
        A01 = [[0 for j in range(
            0, matHalfB)] for i in range(0, matHalfA)]
        A10 = [[0 for j in range(
            0, matHalfB)] for i in range(0, matHalfA)]
        A11 = [[0 for j in range(
            0, matHalfB)] for i in range(0, matHalfA)]

        # Assigning value to smaller lists
        for i in range(0, matHalfA):
            for j in range(0, matHalfB):
                A00[i][j] = A[i][j]
                A01[i][j] = A[i][j + matHalfB]
                A10[i][j] = A[i + matHalfA][j]
                A11[i][j] = A[i + matHalfA][j + matHalfA]

        # Creating new list[list] to hold divided parts of main list B
        B00 = [[0 for j in range(
            0, matHalfB)] for i in range(0, matHalfA)]
        B01 = [[0 for j in range(
            0, matHalfB)] for i in range(0, matHalfA)]
        B10 = [[0 for j in range(
            0, matHalfB)] for i in range(0, matHalfA)]
        B11 = [[0 for j in range(
            0, matHalfB)] for i in range(0, matHalfA)]

        for i in range(0, matHalfB):
            for j in range(0, matHalfB):
                B00[i][j] = A[i][j]
                B01[i][j] = A[i][j + matHalfB]
                B10[i][j] = A[i + matHalfB][j]
                B11[i][j] = A[i + matHalfB][j + matHalfB]

        # Calculating M1
        M11 = matAdd(A00, A11)
        M12 = matAdd(B00, B11)
        M1 = square_matrix_multiply_strassens(M11, M12)

        # Calculating M2
        M21 = matAdd(A10, A11)
        M2 = square_matrix_multiply_strassens(M21, B00)

        # Calculating M3
        M32 = matSub(B01, B11)
        M3 = square_matrix_multiply_strassens(A00, M32)

        # Calculating M4
        M42 = matSub(B10, B00)
        M4 = square_matrix_multiply_strassens(A11, M42)

        # Calculating M5
        M51 = matAdd(A00, A01)
        M5 = square_matrix_multiply_strassens(M51, B11)

        # Calculating M6
        M61 = matSub(A10, A00)
        M62 = matAdd(B00, B01)
        M6 = square_matrix_multiply_strassens(M61, M62)

        # Calculating M7
        M71 = matSub(A01, A11)
        M72 = matAdd(B10, B11)
        M7 = square_matrix_multiply_strassens(M71, M72)

        # Calculating final final matrix elems(FME)
        res00 = matSub(matAdd(matAdd(M1, M4), M7), M5)
        res01 = matAdd(M3, M5)
        res10 = matAdd(M2, M4)
        res11 = matSub(matAdd(matAdd(M1, M3), M6), M2)

        # Final Result matrix
        result = [[0 for j in range(
            0, len(B))] for i in range(0, len(A))]

        # Assigning FME to Final Result Matrix
        for i in range(0, matHalfA):
            for j in range(0, matHalfB):
                result[i][j] = res00[i][j]
                result[i][j + matHalfB] = res01[i][j]
                result[i + (matHalfA)][j] = res10[i][j]
                result[i + (matHalfA)][j + (matHalfB)] = res11[i][j]

    else:

        # 2X2 matrix multiply done here using Strassen's algo
        M1 = (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
        M2 = (A[1][0] + A[1][1]) * B[0][0]
        M3 = A[0][0] * (B[0][1] - B[1][1])
        M4 = A[1][1] * (B[1][0] - B[0][0])
        M5 = (A[0][0] + A[0][1]) * B[1][1]
        M6 = (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
        M7 = (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])

        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 + M3 - M2 + M6

        result = [[C00, C01], [C10, C11]]

    return result


# ==============================================================

# Calculate the power of a matrix in O(k)
def power_of_matrix_navie(A, k):

    """
    Return A^k.
    time complexity = O(k)
    """

    # For k>1, each time 1 matrix instance is separated
    # Hence it produces O(k) complexity in recursion
    if k == 1:
        result = A
    else:
        result = square_matrix_multiply_strassens(
                power_of_matrix_navie(A, 1),
                power_of_matrix_navie(A, k-1))

    return result


# ==============================================================
# Calculate the power of a matrix in O(log k)
def power_of_matrix_divide_and_conquer(A, k):

    """
    Return A^k.
    time complexity = O(log k)
    """

    # k is divided into 2, and both parts multiplied recursively
    # Hence it generates O(log k) complexity
    if k == 1:
        result = A
    else:
        h1 = math.floor(k/2)
        h2 = k - h1
        result = square_matrix_multiply_strassens(
                power_of_matrix_divide_and_conquer(A, h1),
                power_of_matrix_divide_and_conquer(A, h2))

    return result


# ==============================================================
def test():

    assert(find_significant_energy_increase_brute(ENERGY_LEVEL) == (7, 11))

    assert(find_significant_energy_increase_recursive(ENERGY_LEVEL) == (7, 11))

    assert(find_significant_energy_increase_iterative(ENERGY_LEVEL) == (7, 11))

    assert((square_matrix_multiply_strassens([[0, 1], [1, 1]],
                                            [[0, 1], [1, 1]]) ==
                                            asarray([[1, 1], [1, 2]])).all())

    assert((power_of_matrix_navie([[0, 1], [1, 1]], 3) ==
                                  asarray([[1, 2], [2, 3]])).all())

    assert((power_of_matrix_divide_and_conquer([[0, 1], [1, 1]], 3) ==
                                               asarray([[1, 2], [2, 3]])).all())

    # Problem 1: Test multiple scenarios
    print("Problem 1: Test multiple scenarios")
    print(" ")
    # Scenario 1: Given input 1
    print("Scenario 1: Begin - For Given input 1")
    inputEnergyLevel = [100, 113, 110, 85, 105, 102, 86, 63, 81, 101, 94, 106, 101, 79, 94, 90, 97]
    print("Energy Level: %s" % inputEnergyLevel)

    resultB = find_significant_energy_increase_brute(inputEnergyLevel)
    print("BruteForce Result: %s" % (resultB,))

    resultR = find_significant_energy_increase_recursive(inputEnergyLevel)
    print("Recursive Result: %s" % (resultR,))

    resultI = find_significant_energy_increase_iterative(inputEnergyLevel)
    print("Iterative Result: %s" % (resultI,))
    print("Scenario 1: Ends")
    # Scenario 1: Ends
    print(" ")
    # Scenario 2: Given input 2
    print("Scenario 2: Begins - Given input 2")
    inputEnergyLevel = [110, 109, 107, 104, 100]
    print("Energy Level: %s" % inputEnergyLevel)

    resultB = find_significant_energy_increase_brute(inputEnergyLevel)
    print("BruteForce Result: %s" % (resultB,))

    resultR = find_significant_energy_increase_recursive(inputEnergyLevel)
    print("Recursive Result: %s" % (resultR,))

    resultI = find_significant_energy_increase_iterative(inputEnergyLevel)
    print("Iterative Result: %s" % (resultI,))
    print("Scenario 2: Ends")
    # Scenario 2: Ends
    print(" ")
    # Scenario 3: introduced
    print("Scenario 3: introduced- position 5 changed")
    inputEnergyLevel = [100, 113, 110, 85, 185, 102, 86, 63, 81, 101, 94, 106, 101, 79, 94, 90, 97]
    print("Energy Level: %s" % inputEnergyLevel)

    resultB = find_significant_energy_increase_brute(inputEnergyLevel)
    print("BruteForce Result: %s" % (resultB,))

    resultR = find_significant_energy_increase_recursive(inputEnergyLevel)
    print("Recursive Result: %s" % (resultR,))

    resultI = find_significant_energy_increase_iterative(inputEnergyLevel)
    print("Iterative Result: %s" % (resultI,))
    print("Scenario 3: Ends")
    # Scenario 3: Ends
    print(" ")
    print(" ")
    # Problem 2
    # Testing square_matrix_multiply_strassens
    print("Testing square_matrix_multiply_strassens with 2X2 matrix")
    matrix1 = [[0, 1], [1, 1]]
    print("Input matrix: %s" % matrix1)
    result1 = square_matrix_multiply_strassens(matrix1, matrix1)
    print("Result: %s" % result1)
    print(" ")
    print("Testing square_matrix_multiply_strassens with 4X4 matrix")
    matrix1 = [[0, 1, 1, 2], [1, 1, 3, 5], [0, 2, 1, 3], [3, 2, 1, 0]]
    print("Input matrix: %s" % matrix1)
    result1 = square_matrix_multiply_strassens(matrix1, matrix1)
    print("Result: %s" % result1)
    print(" ")
    print("Testing power_of_matrix_navie for k=2")
    matrix1 = [[0, 1], [1, 1]]
    print("Input matrix: %s" % matrix1)
    result1 = power_of_matrix_navie(matrix1, 2)
    print("Result: %s" % result1)
    print(" ")
    print("Testing power_of_matrix_navie for k=3")
    matrix1 = [[0, 1], [1, 1]]
    print("Input matrix: %s" % matrix1)
    result1 = power_of_matrix_navie(matrix1, 3)
    print("Result: %s" % result1)
    print(" ")
    print("Testing power_of_matrix_divide_and_conquer with k=2 matrix")
    matrix1 = [[0, 1], [1, 1]]
    print("Input matrix: %s" % matrix1)
    result1 = power_of_matrix_divide_and_conquer(matrix1, 2)
    print("Result: %s" % result1)
    print(" ")
    print("Testing power_of_matrix_divide_and_conquer with k=3 matrix")
    matrix1 = [[0, 1], [1, 1]]
    print("Input matrix: %s" % matrix1)
    result1 = power_of_matrix_divide_and_conquer(matrix1, 3)
    print("Result: %s" % result1)


if __name__ == '__main__':

    test()

# ==============================================================
