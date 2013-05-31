import sys, math
from math import log
from pprint import pprint

infile = sys.stdin

class Logistic(object):
    # Yeah, I'm doing this by hand
    def train(self, X, y, lambda_val = 1.0):
        self.X = X
        self.y = y
        self.lambda_val = lambda_val

        theta = [0] * len(X[0])
        cost, grad = self._cost(theta)
        iter_num = 1
        alpha = 11
        max_iter = 100
        while True:
            for i in xrange(len(theta)):
                theta[i] = theta[i] - alpha * grad[i]
            next_cost, grad = self._cost(theta)
            sys.stderr.write("iter %d - %f\n" % (iter_num, next_cost))
            if next_cost > cost:
                sys.stderr.write("cost is increasing, use smaller lambda")
                self.theta = theta
                break
            if cost - next_cost < 0.0000000001 or iter_num == max_iter:
                self.theta = theta
                break

            cost = next_cost
            iter_num += 1

    def predict(self, X_query):
        y_query = [0] * len(X_query)
        for i, x in enumerate(X_query):
            h = self.h(self.theta, x)
            if h >= 0.5:
                y_query[i] = 1
            else:
                y_query[i] = 0
        return y_query

    def h(self, theta, x):
        s = sum([x[i] * theta[i] for i in xrange(len(x))])
        return sigmoid(s)

    def _cost(self, theta):
        cost = 0
        grad = [0] * len(theta)
        m = len(self.X)
        for i, xi in enumerate(self.X):
            yi = self.y[i]
            h = self.h(theta, xi)
            cost += (-yi * log(h) - (1 - yi) * log(1-h))
            for j in xrange(len(theta)):
                grad[j] += (h - yi) * xi[j] / m
        cost /= m

        # Add regularization
        cost += (self.lambda_val / (2.0 * m)) * sum([item ** 2 for item in theta[1:]])
        for j in xrange(1, len(theta)):
            grad[j] += self.lambda_val / m * theta[j]

        return cost, grad
        
def main():
    reader = Reader()
    rec_num, feature_num, X, y, query_num, X_query, X_query_id = reader.read()
    #pprint(locals())
    classifier = Logistic()
    classifier.train(X, y)
    y_query = classifier.predict(X_query)
    for i in xrange(len(y_query)):
        y_query[i] = y_query[i] * 2 - 1
    # remove below code before submit()
    # y_train = classifier.predict(X)
    # train_right = sum([1 if y_train[i] == y[i] else 0 for i in xrange(len(y))])
    # sys.stderr.write("train: %d / %d = %f\n" % (train_right, len(y), float(train_right) / len(y)))
    # check(X_query_id, y_query)


    for i in xrange(query_num):
        print "%s %+d" % (X_query_id[i], y_query[i])

def check(X_query_id, y_query):
    idx = 0
    right = 0
    with open("output00.txt") as answer_file:
        line = answer_file.readline()
        while line:
            key, val = line.split()
            val = int(val)
            #sys.stderr.write('%d %d %d\n' % (idx, y_query[idx], val))
            if val == y_query[idx]:
                right += 1
            idx += 1
            line = answer_file.readline()

    sys.stderr.write('%d / %d = %f\n' % (right, len(y_query), float(right) / len(y_query)))


class Reader(object):
    def read(self):
        rec_num, feature_num = [int(item) for item in infile.readline().split()]
        y = list()
        X = list()
        for i in xrange(rec_num):
            rec = [0] * feature_num
            line = infile.readline().split()
            yi = 1 if int(line[1]) == 1 else 0
            y.append(yi)

            line = line[2:]
            for item in line:
                key, value = item.split(':')
                key = int(key) - 1
                value = float(value)
                rec[key] = value
            X.append(rec)

        X_query = list()
        X_query_id = list()
        query_num = int(infile.readline())
        for i in xrange(query_num):
            line = infile.readline().split()
            X_query_id.append(line[0])
            line = line[1:]
            rec = [0] * feature_num
            for item in line:
                key, value = item.split(':')
                key = int(key) - 1
                value = float(value)
                rec[key] = value
            X_query.append(rec)

        X, X_query = self._normalize(X, X_query)

        return rec_num, len(X[0]), X, y, query_num, X_query, X_query_id

    def _normalize(self, X, X_query):
        feature_num = len(X[0])
        rec_num = len(X)
        X_range = [max([line[i] for line in X]) - min([line[i] for line in X]) for i in xrange(feature_num)]
        X_mean =[sum([line[i] for line in X]) / rec_num for i in xrange(feature_num)]
        X_norm = list()
        X_query_norm = list()

        for i in xrange(rec_num):
            X_norm.append([1,])
            for j in xrange(feature_num):
                if X_range[j] != 0:
                    X_norm[i].append((X[i][j] - X_mean[j]) / X_range[j])
        for i in xrange(len(X_query)):
            X_query_norm.append([1,])
            for j in xrange(feature_num):
                if X_range[j] != 0:
                    X_query_norm[i].append((X_query[i][j] - X_mean[j]) / X_range[j])
        return X_norm, X_query_norm

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

if __name__ == "__main__":
    main()
