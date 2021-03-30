# UNIDAD 3: MÍNIMOS CUADRADOS
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    # Ajuste de datos
    m = len(t)
    A = [[t[i]**j for j in range(n)] for i in range(m)]

    # Transpuestas y descomposición de Cholesky
    AT = np.transpose(A)
    A = np.matmul(AT, A)
    L = np.linalg.cholesky(A)
    LT = np.transpose(L)

    # Solución de los sistemas triangulares
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

    # Ajuste de datos
    m = len(t)
    A = [[t[i]**j for j in range(n)] for i in range(m)]
    b = [i for i in y]

    for i in range(n):

        # Obtención de a y alfa
        a = [A[j][i] for j in range(m)]
        alfa = 0
        for j in a: alfa += j**2
        alfa **= 0.5
        if (A[i][i] > 0): alfa *= -1

        # Cómputo de v
        v = [0 for _ in range(i)]
        for j in range(i, m):
            if (j == i): v += [a[j] - alfa]
            else: v += [a[j]]

        # Cómputo de H en A
        for k in range(n):
            vTx, vTv = 0, 0
            for j in range(m): vTx += v[j] * A[j][k]
            for j in range(m): vTv += v[j] * v[j]
            for j in range(i, m): A[j][k] = A[j][k] - 2 * (vTx/vTv) * v[j]

        # Cómputo de H en b
        vTx, vTv = 0, 0
        for j in range(m): vTx += v[j] * b[j]
        for j in range(m): vTv += v[j] * v[j]
        for j in range(i, m): b[j] = b[j] - 2 * (vTx/vTv) * v[j]

    x = sucesiva_hacia_atras(A[:n], b[:n])

    return x


# Ejecuta una función polinómica para una entrada t con parámetros x
def polinomio(n, t, x):
    ft = 0
    for i in range(n): ft += x[i]*t**i
    return ft


# Dibuja los puntos y función resultante en el plano
def ejemplo(n, te, ye, tv, yv, metodo):

    # Obtención de x
    if (metodo == 1):
        start = time.time()
        x = ecuaciones_normales(n, te, ye)
        time_ = time.time() - start
    else:
        start = time.time()
        x = householder(n, te, ye)
        time_ = time.time() - start
    
    # Gráfica de la función
    me, mv = len(te), len(tv)
    min_t, max_t = min(te), max(te)
    t_funcion = np.linspace(min_t, max_t, 1000)
    y_funcion = [polinomio(n, i, x) for i in t_funcion]
    plt.plot(t_funcion, y_funcion, color="black")

    # Exactitud del método (Usando el Error Cuadrático Medio)
    yp = [polinomio(n, i, x) for i in tv]
    ecm = 0
    for i in range(mv): ecm += (yp[i] - yv[i])**2
    ecm /= mv

    # Resultado de x y tiempo de ejecución
    if (metodo == 1): plt.title("Método de Ecuaciones Normales")
    else: plt.title("Método de Householder")
    print("x =", [round(i, 3) for i in x])
    print("Tiempo =", round(time_, 5))
    print("ECM =", round(ecm, 3))

    # Gráfica de los conjuntos de entrenamiento y validación
    for i in range(me): plt.plot(te[i], ye[i], marker="o", markersize=5, color="blue")
    for i in range(mv): plt.plot(tv[i], yv[i], marker="o", markersize=5, color="green")
    for i in range(mv): plt.plot(tv[i], yp[i], marker="o", markersize=5, color="red")
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.show()


# EJEMPLO DE PRUEBA (También se encuentra en el informe)
def main():

    # Importación de los datos de prueba
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
    datos = pd.read_csv(url)
    datos["Month"] = datos["Month"].str.replace('-','').astype(float)


    # Separación de los datos de entrenamiento y validación
    t, y = datos.drop("Sales", axis=1), datos["Sales"]
    te, tv, ye, yv = train_test_split(t, y, test_size=0.3, random_state=42)
    te = te["Month"].values.tolist()
    ye = ye.tolist()
    tv = tv["Month"].values.tolist()
    yv = yv.tolist()

    # Solución con ambos métodos
    ejemplo(3, te, ye, tv, yv, 1)
    ejemplo(3, te, ye, tv, yv, 2)


main()