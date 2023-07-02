# pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
# pairs.sort(key=lambda pair: pair[1])
# print(pairs)
matrix = [
     [1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12], ]
print([[row[i] for row in matrix] for i in range(4)])
print(matrix)