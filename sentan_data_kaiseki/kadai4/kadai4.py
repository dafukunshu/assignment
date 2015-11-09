import numpy as np
import matplotlib.pyplot as plt

n = 10
N = 2000

class Model1:
  def __init__(self):
    self.theta = np.ones(2)
  def f(self, x):
    return (self.theta[0] + self.theta[1:] * x)

def huber_w(r):
  eta = 1
  w = np.ndarray(r.shape)
  w[r <= eta] = 1
  w[r > eta] = eta / np.fabs(r[r > eta])
  return w

def tukey_w(r):
  eta = 1
  w = np.ndarray(r.shape)
  w[r <= eta] = (1 - r[r <= eta] ** 2 / eta ** 2) ** 2
  w[r > eta] = 0
  return w

def train_model(x, y, calc_w):
  model = Model1()

  Phi = np.ndarray((n, 2))
  Phi[:, 0] = np.ones(n)
  Phi[:, 1] = x

  i = 0
  while True:
    i += 1
    r = np.fabs(model.f(x) - y)
    w = calc_w(r)
    
    W = np.diag(w)
    theta = np.linalg.inv(Phi.T.dot(W.dot(Phi))).dot(Phi.T.dot(W.dot(y.reshape(len(y), 1))))
    old_theta = model.theta
    model.theta = theta.flatten()
    if np.linalg.norm(old_theta - model.theta) < 0.001:
      break

  print "last_iter = %d" % i
  return model

if __name__ == "__main__":
  np.random.seed(0)

  # make dataset
  x = np.linspace(-3, 3, n)
  y = x + 0.2 * np.random.randn(n)
  y[-1] = -4
  y[-2] = -4

  # get trained model
  print "huber"
  model1 = train_model(x, y, huber_w)
  print "tukey"
  model2 = train_model(x, y, tukey_w)
  
  # test & plot
  X= np.linspace(-4, 4, N)
  Y = model1.f(X)
  plt.plot(X, Y, label="huber")
  Y = model2.f(X)
  plt.plot(X, Y, label="tukey")
  plt.plot(x, y, "o")
  plt.legend(loc="lower right")
  plt.show()
