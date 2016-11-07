import numpy as np
from matplotlib import pyplot as plt

class RNN(object):

    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden_weight = np.random.randn(n_hidden, n_input + 1)
        self.output_weight = np.random.randn(n_output, n_hidden + 1)
        self.recurr_weight = np.random.randn(n_hidden, n_hidden + 1)


    def train(self, Xl, epsilon, lam, epoch):
        self.__loss = np.zeros(epoch)
        for epo in range(epoch):
            print 'epoch: {0}'.format(epo)
            for X in np.random.permutation(Xl):
                tau = X.shape[0]
                zs, ys = self.__forward_seq(X)
                hidden_delta = np.zeros(self.n_hidden)
                output_dEdw = np.zeros(self.output_weight.shape)
                hidden_dEdw = np.zeros(self.hidden_weight.shape)
                recurr_dEdw = np.zeros(self.recurr_weight.shape)

                for t in range(tau - 1)[::-1]:

                    # output delta
                    output_delta = (ys[t] - X[t + 1, :]) * (1.0 - ys[t] ** 2)
                    output_dEdw += output_delta.reshape(-1, 1) * np.hstack((1.0, zs[t]))

                    # hidden delta
                    hidden_delta = (self.output_weight[:, 1:].T.dot(output_delta) + self.recurr_weight[:, 1:].T.dot(hidden_delta)) * zs[t] * (1.0 - zs[t])
                    hidden_dEdw += hidden_delta.reshape(-1, 1) * np.hstack((1.0, X[t, :]))

                    # recurr delta
                    zs_prev = zs[t - 1] if t > 0 else np.zeros(self.n_hidden)
                    recurr_dEdw += hidden_delta.reshape(-1, 1) * np.hstack((1.0, zs_prev))

                    # accumulate loss
                    self.__loss[epo] += 0.5 * (ys[t] - X[t + 1]).dot((ys[t] - X[t + 1]).reshape((-1, 1))) / (tau - 1)

                # update weights
                self.output_weight -= epsilon * (output_dEdw + lam * self.output_weight)
                self.hidden_weight -= epsilon * hidden_dEdw
                self.recurr_weight -= epsilon * recurr_dEdw


    def save_param(self, fn = 'weights.npy'):
        weights = {'h': self.hidden_weight, 'o': self.output_weight, 'r': self.recurr_weight}
        np.save(fn, weights)


    def save_lossfig(self, fn = 'loss.png'):
        plt.plot(np.arange(self.__loss.size), self.__loss)
        plt.savefig(fn)


    @classmethod
    def load(cls, fn = 'weights.npy'):
        weights = np.load(fn).item()
        n_input = weights['h'].shape[1] - 1
        n_hidden = weights['h'].shape[0]
        n_output = weights['o'].shape[0]
        rnn = RNN(n_input, n_hidden, n_output)
        rnn.hidden_weight = weights['h']
        rnn.output_weight = weights['o']
        rnn.recurr_weight = weights['r']
        return rnn


    def predict(self, X):
        _, ys = self.__forward_seq(X)
        return ys


    def predict_loop(self, X, times):
        zs, ys = self.__forward_seq(X)
        y, z = ys[-1], zs[-1]
        for i in range(times):
            z, y = self.__forward(y, z)
            zs.append(z)
            ys.append(y)

        return ys


    def __sigmoid(self, arr):
        return 1.0 / (1.0 + np.exp(-arr))


    def __tanh(self, arr):
        pl = np.exp(arr)
        mn = np.exp(-arr)
        return (pl - mn) / (pl + mn)


    def __forward(self, x, z):
        r = self.recurr_weight.dot(np.hstack((1.0, z)))
        z = self.__sigmoid(self.hidden_weight.dot(np.hstack((1.0, x))) + r)
        y = self.__tanh(self.output_weight.dot(np.hstack((1.0, z))))
        return (z, y)


    def __forward_seq(self, X):
        z = np.zeros(self.n_hidden)
        zs, ys = ([], [])
        for x in X:
            z, y = self.__forward(x, z)
            zs.append(z)
            ys.append(y)
        return zs, ys
