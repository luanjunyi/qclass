import sys
from pprint import pprint

infile = sys.stdin

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
            X_norm.append(list())
            for j in xrange(feature_num):
                if X_range[j] != 0:
                    X_norm[i].append((X[i][j] - X_mean[j]) / X_range[j])

        for i in xrange(len(X_query)):
            X_query_norm.append(list())
            for j in xrange(feature_num):
                if X_range[j] != 0:
                    X_query_norm[i].append((X_query[i][j] - X_mean[j]) / X_range[j])

        return X_norm, X_query_norm


def main():
    reader = Reader()
    rec_num, feature_num, X, y, query_num, X_query, X_query_id = reader.read()
    pprint(locals())
        

if __name__ == "__main__":
    main()
