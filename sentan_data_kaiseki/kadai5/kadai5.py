import numpy as np
import cPickle as pkl
from scipy.io import loadmat

np.random.seed(0)
n_labels = 10
def gauss_kernel(x, c):
  h = 2
  if len(x.shape) == 2:
    y = np.exp(-np.linalg.norm(x-c, axis=1) ** 2 / ( 2 * (h ** 2)))
  else:
    y = np.exp(-np.dot(x-c, (x-c).T) / ( 2 * (h ** 2)))
  return y

class KernelModel:
  def __init__(self, X, y, kernel):
    self.X = X
    self.y = y
    self.w = np.random.uniform(-1, 1, len(self.X))
    self.kernel = kernel
  def f(self, x):
    xs = np.tile(x, (len(self.X), 1))
    y = np.dot(self.w, self.kernel(xs, self.X))
    return y
   
def load_dataset():
  data = loadmat("digit.mat")
  x_train = data["X"]
  x_test = data["T"]
  n_train_class_samples = 500
  n_test_class_samples = 200
  n_train_samples = n_train_class_samples * n_labels
  n_test_samples = n_test_class_samples * n_labels
  image_size = 256
  
  x_train = x_train.transpose(2, 1, 0)
  x_test = x_test.transpose(2, 1, 0)
  x_train = x_train.reshape(n_train_samples, image_size)
  x_test = x_test.reshape(n_test_samples, image_size)

  y_train = np.zeros(n_train_class_samples * n_labels).astype("int")
  y_test = np.zeros(n_test_class_samples * n_labels).astype("int")
  for i in xrange(n_labels):
    y_train[i * n_train_class_samples:(i+1) * n_train_class_samples] = i
    y_test[i * n_test_class_samples:(i+1) * n_test_class_samples] = i

  return x_train, y_train, x_test, y_test

def kernel_matrix(X, kernel):
  Kmat = np.ndarray((len(X), len(X)))
  for i in xrange(len(X)):
    #if i % 1000 == 0:
    #  print i
    Xis = np.tile(X[i], (len(X), 1))
    Kmat[i, :] = kernel(Xis, X)
    #for j in xrange(len(X)):
    #  Kmat[i, j] = kernel(X[i], X[j])
  return Kmat

def train(model):
  lamb = 0.1
  Kmat = kernel_matrix(model.X, model.kernel)
  #Kmat_inv = np.linalg.inv(Kmat)
  #model.w = np.dot(Kmat_inv, model.y)
  model.w = np.linalg.inv(Kmat.dot(Kmat) + lamb * np.eye(len(Kmat))).dot(Kmat.T.dot(model.y))
  return model

def ovr_train(x_train, y_train):
  models = np.ndarray(n_labels, dtype=object)
  for i in xrange(n_labels):
    yy_train = np.copy(y_train)
    yy_train[y_train == i] = 1
    yy_train[y_train != i] = -1
    model = KernelModel(x_train, yy_train, gauss_kernel)
    models[i] = train(model)
  return models

def ovo_train(x_train, y_train):
  models = np.ndarray((n_labels, n_labels), dtype=object)
  for i in xrange(n_labels):
    for j in xrange(i+1, n_labels):
      xx_train = np.concatenate((x_train[y_train == i], x_train[y_train == j]))
      yy_train = np.copy(np.concatenate((y_train[y_train == i], y_train[y_train == j])))
      yyy_train = np.copy(yy_train)
      yyy_train[yy_train == i] = 1
      yyy_train[yy_train == j] = -1
      model = KernelModel(xx_train, yyy_train, gauss_kernel)
      models[i, j] = train(model)
  return models

def train_models(x_train, y_train, strategy):
  if strategy == "ovr":
    models = ovr_train(x_train, y_train)
  else:
    assert strategy == "ovo"
    models = ovo_train(x_train, y_train)

  return models

def test(model, X):
  y = np.asarray([model.f(x) for x in X])
  return y

def make_confuse_matrix(y_preds, y_truths):
  conmat = np.zeros((n_labels, n_labels)).astype("int")
  for y_pred, y_truth in zip(y_preds, y_truths):
    conmat[y_truth][y_pred] += 1
  return conmat

def ovr_evaluate(models, x_test, y_test):
  scores = np.ndarray((n_labels, len(x_test)), dtype="float")
  for i in xrange(n_labels):
    scores[i] = test(models[i], x_test)  
  y_pred = np.argmax(scores, axis=0)
  conmat = make_confuse_matrix(y_pred, y_test)
  return conmat

def ovo_evaluate(models, x_test, y_test):
  y_pred = np.ndarray(len(x_test), dtype="int")
  for x_i in xrange(len(x_test)):
    vote = np.zeros(n_labels).astype("int")
    for i in xrange(n_labels):
      for j in xrange(i+1, n_labels):
        score = np.sign(models[i][j].f(x_test[x_i]))
        if score > 0:
          vote[i] += 1
        elif score < 0:
          vote[j] += 1
    y_pred[x_i] = np.argmax(vote)

  conmat = make_confuse_matrix(y_pred, y_test)
  return conmat

def evaluate(models, x_test, y_test, strategy):
  if strategy == "ovr":
    conmat = ovr_evaluate(models, x_test, y_test)
  else:
    assert strategy == "ovo"
    conmat = ovo_evaluate(models, x_test, y_test)
  return conmat

def run(strategy):
  x_train, y_train, x_test, y_test = load_dataset()
  models = train_models(x_train, y_train, strategy)
  #np.save("models.npy", models)
  print evaluate(models, x_train, y_train, strategy)
  print evaluate(models, x_test, y_test, strategy)

if __name__ == "__main__":
  run("ovr")
  run("ovo")
