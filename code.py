# UNIDAD 3: MÍNIMOS CUADRADOS
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Funciones auxiliares del anterior laboratorio
def sucesiva_hacia_atras(A, b):
    """
    Entrada: una matriz triangular superior A y un vector b.
    Salida: un vector x tal que Ax = b.
    """
    if (np.linalg.det(A) == 0):
        print("A es una matriz singular, el sistema no tiene solución.")
        return []
    else:
        n = len(A) - 1
        x = [None for _ in range(n)] + [b[n] / A[n][n]]
        for i in range(n, -1, -1):
            sumatoria = 0
            for j in range(i+1, n+1):
                sumatoria += A[i][j] * x[j]
            x[i] = round((b[i] - sumatoria) / A[i][i], 5)

        return x


def sucesiva_hacia_adelante(A, b):
    """
    Entrada: una matriz triangular inferior A y un vector b.
    Salida: un vector x tal que Ax = b.
    """
    if (np.linalg.det(A) == 0):
        print("A es una matriz singular, el sistema no tiene solución.")
        return []
    else:
        n = len(A) - 1
        x = [b[0] / A[0][0]] + [None for _ in range(n)]
        for i in range(1, n+1):
            sumatoria = 0
            for j in range(i):
                sumatoria += A[i][j] * x[j]
            x[i] = round((b[i] - sumatoria) / A[i][i], 5)

        return x


# Método de Ecuaciones Normales
def ecuaciones_normales(n, t, y):
    """
    Entrada: un entero n, un vector t y un vector y.
    Salida: un vector x de parámetros para un ajuste polinomial (orden n) 
            usando los datos de t (entrada) & y (salida).
    """

    m = len(t)
    A = [[t[i]**j for j in range(n)] for i in range(m)]

    AT = np.transpose(A)
    A = np.matmul(AT, A)
    L = np.linalg.cholesky(A)
    LT = np.transpose(L)

    ye = sucesiva_hacia_adelante(L, np.matmul(AT, y))
    x = sucesiva_hacia_atras(LT, ye)

    return x


# Método de Householder
def householder(n, t, y):
    """
    Entrada: un entero n, un vector t y un vector y.
    Salida: un vector x de parámetros para un ajuste polinomial (orden n) 
            usando los datos de t (entrada) & y (salida).
    """

    m = len(t)
    A = [[t[i]**j for j in range(n)] for i in range(m)]
    b = [i for i in y]

    for i in range(n):

        a = [A[j][i] for j in range(m)]
        alfa = 0
        for j in a: alfa += j**2
        alfa = alfa**0.5

        if (A[i][i] > 0): alfa *= -1
        v = [0 for _ in range(i)]
        for j in range(i, m):
            if (j == i): v += [a[j] - alfa]
            else: v += [a[j]]

        for k in range(n):
            vTx, vTv = 0, 0
            for j in range(m): vTx += v[j] * A[j][k]
            for j in range(m): vTv += v[j] * v[j]
            for j in range(i, m): A[j][k] = A[j][k] - 2 * (vTx/vTv) * v[j]
        
        vTx, vTv = 0, 0
        for j in range(m): vTx += v[j] * b[j]
        for j in range(m): vTv += v[j] * v[j]
        for j in range(i, m): b[j] = b[j] - 2 * (vTx/vTv) * v[j]

    x = sucesiva_hacia_atras(A[:n], b[:n])

    return x


# Ejecuta una función polinómica
def polinomio(n, t, x):
    """
    Entrada: un entero n, un entero t y un vector x.
    Salida: un entero ft que resulta de ejecutar la función polinómica 
            de orden n con los parámetros de x.
    """
    ft = 0
    for i in range(n): ft += x[i]*t**i
    return ft


# Dibuja los puntos y función resultante en el plano
def paint_example(n, t, y):
    m, min_t, max_t = len(t), min(t), max(t)
    t_ = np.linspace(min_t, max_t, 1000)

    start = time.time()
    xe = ecuaciones_normales(n, t, y)
    e_time = time.time() - start
    ye = [polinomio(n, i, xe) for i in t_]
    print("Método de Ecuaciones Normales")
    print("x = {0}".format([round(i, 3) for i in xe]))
    print("Tiempo = {0}".format(round(e_time, 5)))
    for i in range(m): plt.plot(t[i], y[i], marker="o", color="black")
    plt.plot(t_, ye, color="blue")
    plt.grid()
    plt.show()

    start = time.time()
    xh = householder(n, t, y)
    h_time = time.time() - start
    yh = [polinomio(n, i, xh) for i in t_]
    print("Método de Householder")
    print("x = {0}".format([round(i, 5) for i in xh]))
    print("Tiempo = {0}".format(round(h_time, 5)))
    for i in range(m): plt.plot(t[i], y[i], marker="o", color="black")
    plt.plot(t_, yh, color="red")
    plt.grid()
    plt.show()


# EJEMPLOS DE PRUEBA (También se encuentran en el informe)
def main():
    n = 3
    t = [-1, -0.5, 0, 0.5, 1]
    y = [1, 0.5, 0, 0.5, 2]
    paint_example(n, t, y)


main()