import numpy
from math import *
from random import *
import csv
import pandas as pd


# --------- set parameters ----------
N = 10
M = 9
alpha = 0.005
belta = 11.1


def phi(x):
    ans = [[x ** i] for i in range(0, M + 1)]

    return numpy.array(ans)
    #return numpy.array([[x ** i] for i in range(M + 1)])

def cal_S(trainData):
    I = numpy.identity(M + 1)
    sumOfphi = numpy.sum([phi(x).dot(phi(x).T) for x in trainData], axis=0)
    S_inverse = alpha * I + belta * sumOfphi
    return numpy.linalg.inv(S_inverse)


def cal_mean(x, trainData, targetData):
    sum_phi_times_t = numpy.sum([phi(x) * t for x, t in zip(trainData, targetData)], axis=0)
    return belta * (phi(x).T).dot(cal_S(trainData).dot(sum_phi_times_t))[0][0]

def cal_var(x, trainData):
    return (belta ** -1 + (phi(x).T).dot(cal_S(trainData).dot(phi(x))))[0][0]



def predict(X, T, newX):
    print('In ' + str(newX) + ' day '  + '  the prediction distribution is:')
    print('The distribution predicted with mean : ' + str(cal_mean(newX, X, T)))
    print('The distribution predicted with variance : ' + str(cal_var(newX, X)))


def readCSV():
    myCSV = pd.read_csv('GE.csv')
    return list(myCSV.Low[:10])


def performance_by_test_ten_samples(company):

    for cpy in company:
        filename = cpy + '.csv'
        csvFile = pd.read_csv(filename)
        T = csvFile.High[: N]
        realT = csvFile.High[N]

        X = [i for i in range(1, N + 1)]
        abs_error = abs(cal_mean(N, X, T) - realT)
        relative_error = abs(cal_mean(N, X, T) - realT) / realT

        print('------- the performance of predicting the 11th day High price of ' + cpy + ' --------')
        print('absolute mean error is : ' + str(abs_error))
        print('relative error is : ' + str(relative_error))


# ------------ performance evaluation ------------
Company = ['AAPL', 'AMZN', 'BABA', 'BE', 'D', 'GE', 'IBM', 'MU', 'TME', 'TSLA']
performance_by_test_ten_samples(Company)









