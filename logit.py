import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn import metrics
from logit_model import Logit

periods = [3, 10, 15, 30, 60, 90, 180, 365]
print("Periods:", periods)
res_in = {k:[] for k in range(1, 11)}
res_out = {k:[] for k in range(1, 11)}
for dataset in range(1, 11):
    res = []
    file = "data" + str(dataset) + ".csv"
    with open(file, 'r') as f:
        data = csv.reader(f, delimiter=',')
        values = [i[1] for i in data][1:]
    for period in periods:

        X = []
        y = []
        for i in range(365, len(values) - period):
            X.append([float(j) for j in values[(i - 365):i]])
            if values[i] < values[i + period]:
                y.append(1)
            else:
                y.append(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

        lr = Logit()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        y_pred_t = lr.predict(X_train)

        # out of sample
        res_out[dataset].append(str(round(metrics.accuracy_score(y_test, y_pred) * 100, 1)) + "%")
        # in sample
        res_in[dataset].append(str(round(metrics.accuracy_score(y_train, y_pred_t) * 100, 1)) + "%")

    print(dataset)

print("Accuracy in sample")
print("\n".join(["data" + str(k) + "\t" + "\t".join(res_in[k]) for k in range(1, 11)]))

print("Accuracy out sample")
print("\n".join(["data" + str(k) + "\t" + "\t".join(res_out[k]) for k in range(1, 11)]))
