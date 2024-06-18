import numpy as np


def r1(t):
    r = [[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]]
    return r


def r2(t):
    r = [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]
    return r


def r3(t):
    r = [[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]]
    return r


class Arm:
    def __init__(self, l1, l2, l3, ee, angles_i, inputs):
        self.link2 = None
        self.link1 = None
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.ee = ee
        self.angle = angles_i
        self.inputs = inputs

    def position(self):
        j1p = [0, 0, 0]
        rot1 = r2(self.angle[0])
        rot2 = np.matmul(r3(self.angle[1]), rot1)
        rot3 = np.matmul(r3(self.angle[2]), rot2)
        rot4 = np.matmul(np.matmul(np.matmul(r1(self.angle[5]),r3(self.angle[4])),r1(self.angle[3])),rot3)
        j2p = j1p + np.matmul(rot1, np.array(self.l1))
        j3p = j2p + np.matmul(rot2, np.array(self.l2))
        j4p = j3p + np.matmul(rot3, np.array(self.l3))
        j5p = j4p + np.matmul(rot4, np.array(self.ee))

        self.link1 = [[[j1p[0]], [j2p[0]]], [[j1p[1]], [j2p[1]]], [[j1p[2]], [j2p[2]]]]
        self.link2 = [[[j2p[0]], [j3p[0]]], [[j3p[1]], [j3p[1]]], [[j2p[2]], [j3p[2]]]]
        self.link3 = [[[j3p[0]], [j4p[0]]], [[j3p[1]], [j4p[1]]], [[j3p[2]], [j4p[2]]]]
        self.link4 = [[[j4p[0]], [j5p[0]]], [[j4p[1]], [j5p[1]]], [[j4p[2]], [j5p[2]]]]


        return j1p, j2p, j3p, j4p, j5p


    def transformation(self, i):
        M = self.link1[0][1][0]**2 + self.link2[0][1][0]**2 + self.link3[0][1][0]**2 + self.link4[0][1][0]**2
        V = self.inputs[6,i]
        A = self.inputs[12,i]
        return M,V,A

    def inCreate(self):
        M = np.zeros([np.shape(self.inputs)[1], 1])
        V = np.zeros([np.shape(self.inputs)[1], 1])
        A = np.zeros([np.shape(self.inputs)[1], 1])
        for i in range(np.shape(self.inputs)[1]):
            M[i], V[i], A[i] = self.transformation(i)
        inUsed = [M, V, A]
        return inUsed

    def simulate(self):
        self.angle = self.inputs[0]
        self.position()
        M = []
        V = []
        A = []
        Mi,Vi,Ai = self.transformation(0)
        M.append(Mi)
        V.append(Vi)
        A.append(Ai)
        for i in range(1, np.shape(self.inputs)[1]):
            self.angle = self.inputs[0:6,i]
            self.position()
            Mi, Vi, Ai = self.transformation(i)
            M.append(Mi)
            V.append(Vi)
            A.append(Ai)

        return M, V, A