#coding=utf-8
"""
Implements Information Cell Mixture Model (Tang and Lawry, 2010)
"""
import numpy as np
import scipy.cluster as cluster
import scipy.stats as stats
import matplotlib.pyplot as plt


class ICMM:
    def __init__(self):
        self.P = None
        self.Pr = None # Weights of information cells
        self.N = 0  # length of data set
        self.n = 0  # number of information cells
        self.embedding = None  # input embedding information
        self.c = None
        self.sigma = None
        self.stats = {}

    def set_embedding(self, embedding):
        self.embedding = embedding
        self.N = embedding.shape[0]

    def update_q_ik(self, epsilon):
        q_ik = np.zeros(shape=(self.n, self.N))
        for i in range(self.n):
            F_e_c_sigma = 1 - stats.norm.cdf(0, loc=self.c[i], scale=self.sigma[i])
            for k in range(self.N):
                f_e_c_sigma = stats.norm.pdf(epsilon[i,k], loc=self.c[i], scale=self.sigma[i])
                q_ik[i, k] = self.Pr[i] * f_e_c_sigma / F_e_c_sigma
        sum_over_i = np.sum(q_ik, axis=0)
        for k in range(self.N):
            q_ik[:, k] = q_ik[:, k]/sum_over_i[k]
        return q_ik

    def get_cost(self, epsilon):
        J = 0
        for k in range(self.N):
            subsum = 0
            for i in range(self.n):
                subsum += stats.norm.pdf(epsilon[i,k], loc=self.c[i], scale=self.sigma[i]) * self.Pr[i]
            J += np.log(subsum)
        return J

    def update_icmm(self, num_cells, max_iter=1000, th=1e-5):
        if self.embedding is None:
            raise Exception('Error: Embedding not set.')
        self.n = num_cells

        # (i)Obtain prototypes by k-means
        #obs = cluster.vq.whiten(self.embedding)
        obs = self.embedding
        self.P, self.stats['distortion_kmeans'] = cluster.vq.kmeans2(obs, self.n)
        self.Pr = np.ones(shape=(self.n,)) * (1/float(self.n))    # initial probability distribution of each cell

        # (ii)Compute distances
        epsilon = np.zeros(shape=(self.n, self.N))
        for i in range(self.n):
            # i-th cell
            for k in range(self.N):
                # to k-th observation
                epsilon[i,k] = np.linalg.norm(self.embedding[k,:] - self.P[i,:])

        # (iii)Initialize c_i and sigma_i
        self.c = np.mean(epsilon, axis=1)
        self.sigma = np.std(epsilon, axis=1)

        # (iv)Compute weights
        q_ik = self.update_q_ik(epsilon)

        # (v)Repeat
        J_old = self.get_cost(epsilon)

        for iter in range(max_iter):
            # (b) update probability distribution of information cells
            self.Pr = np.mean(q_ik, axis=1)
            # (c) update density function
            sub_c = np.sum(np.multiply(q_ik, epsilon), axis=1)
            self.c = np.divide(sub_c, np.sum(q_ik, axis=1))
            del sub_c
            temp = epsilon - np.tile(np.array([self.c]).T, (1, self.N))
            temp = np.multiply(np.square(temp), q_ik)
            self.sigma = np.sqrt(np.divide(np.sum(temp, axis=1), np.sum(q_ik, axis=1)))
            del temp
            # (d) compute weights
            q_ik = self.update_q_ik(epsilon)
            J = self.get_cost(epsilon)
            if iter%10 == 0:
                print("iteration %d: J = %.2f" %(iter, J))
            if np.abs(J - J_old) < th:
                print("Final iteration %d: J = %.2f" % (iter, J))
                break
            J_old = J

    def get_mu(self, X):
        mu = 0
        for i in range(self.n):
            d_X_Pi = np.linalg.norm(X - self.P[i])
            mu += self.Pr[i] * (1 - stats.norm.cdf(d_X_Pi, loc=self.c[i], scale=self.sigma[i]))
        return mu


def test():
    mean0 = [-5,0]
    cov0 = [[5,0], [0,5]]
    x1, y1 = np.random.multivariate_normal(mean0, cov0, 500).T
    mean1 = [2,2]
    cov1 = [[2,0], [0,2]]
    x2, y2 = np.random.multivariate_normal(mean1, cov1, 500).T
    plt.scatter(x1, y1, color='r')
    plt.scatter(x2, y2, color='g')
    plt.show()

    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    icmm = ICMM()
    embedding = np.column_stack((x, y))
    cluster.vq.whiten(embedding)
    #plt.scatter(embedding[:,0], embedding[:,1])
    #plt.show()
    icmm.set_embedding(embedding)
    icmm.update_icmm(num_cells=2)

    #Evaluate
    mesh_x = np.arange(-10, 10, 0.1)
    mesh_y = np.arange(-10, 10, 0.1)

    zz = np.zeros((mesh_x.shape[0], mesh_y.shape[0]))
    for i in range(mesh_x.shape[0]):
        for j in range(mesh_y.shape[0]):
            zz[i, j] = icmm.get_mu(np.array([mesh_x[i], mesh_y[j]]))
    # plt.show()
    plt.pcolormesh(mesh_x, mesh_y, zz, vmin=0, vmax=1.0)
    #plt.contourf(mesh_x, mesh_y, zz)
    plt.colorbar()
    plt.show()
    print('done')


if __name__ == '__main__':
    test()
    print('done')
