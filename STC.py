import numpy as np

H0 = np.array([[1, 1], [0, 1]])
H1 = np.array([[1, 0], [1, 1]])

m = np.array([0, 1])

x = np.array([1, 0, 1, 1])

y = x.copy()

def D(x, y):
    return np.sum(x != y)

def S(H, y):
    return np.mod(np.dot(H, y), 2)

class Node:
    def __init__(self, pos, syn, dis, pre):
        self.pos = pos 
        self.syn = syn 
        self.dis = dis 
        self.pre = pre

lattice = [[Node(0, np.array([0, 0]), 0, None)]]
for i in range(2): 
    Hi = H0 if m[i] == 0 else H1 
    for j in range(2): #
        col = [] 
        for k in range(4): 
            syn = np.mod(k, 2) 
            dis = float('inf') 
            pre = None 
            for node in lattice[-1]: 
                yj = 1 - y[2 * i + j] if node.syn[0] == syn else y[2 * i + j] 
                dj = node.dis + D(x[2 * i + j], yj) 
                if dj < dis:
                    dis = dj
                    pre = node
                    y[2 * i + j] = yj
            col.append(Node(2 * i + j + 1, np.array([syn, S(Hi[:, j], y[2 * i: 2 * i + j + 1])]), dis, pre)) 
        lattice.append(col)

end = min(lattice[-1], key=lambda node: node.dis)

path = [end]
while path[-1].pre:
    path.append(path[-1].pre)
path.reverse()

print("The optimal path is:")
for node in path:
    print(f"Position: {node.pos}, Syndrome: {node.syn}, Distortion: {node.dis}")

print(f"The embedded vector y is: {y}")
