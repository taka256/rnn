from rnn import *
from matplotlib import pyplot

def save_loopfig(rnn, st, en, div, m, fn):
    n = int((en - st) / pi * div + 1)
    x = np.linspace(st, en, n)
    ys = rnn.predict_loop(np.sin(x[:m]), n - m)
    ys = np.array(ys)[:, 0]
    pyplot.figure(figsize = (12, 6))
    pyplot.plot(x[1:m], ys[1:m], '--o')
    pyplot.plot(x[m - 1:], ys[m - 1:], '--o')
    pyplot.savefig(fn)
    pyplot.clf()

if __name__ == '__main__':

    N = 7000
    pi = np.pi
    div = 6
    s = (np.random.rand(N) * pi).reshape(-1, 1)
    e = s + np.random.randint(2, 5, N).reshape(-1, 1) * pi / 2
    Xl = [np.linspace(_s, _e, (_e - _s) / pi * div + 1).reshape(-1, 1) for _s, _e in np.hstack((s, e))]
    Xl = map(lambda X: np.sin(X), Xl)

    rnn = RNN(1, 4, 1)
    rnn.train(Xl, epsilon = 0.05, lam = 0.001, epoch = 30)
    rnn.save_param()
    rnn.save_lossfig()

    st = np.random.rand() * pi
    en = st + 20 * pi
    save_loopfig(rnn, st, en, div, m = 7, fn = 'loop.png')
