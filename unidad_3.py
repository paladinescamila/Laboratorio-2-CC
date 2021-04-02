# UNIDAD 3: MÍNIMOS CUADRADOS
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    Salida: un vector x de parámetros para un ajuste polinomial (orden n-1) 
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
    Salida: un vector x de parámetros para un ajuste polinomial (orden n-1) 
            usando los datos de t (entrada) & y (salida).
    """

    m = len(t)
    A = [[t[i]**j for j in range(n)] for i in range(m)]
    b = [i for i in y]

    for i in range(n):

        a = [0 for _ in range(i)] + [A[j][i] for j in range(i, m)]
        alfa = np.linalg.norm(a) if (A[i][i] < 0) else (-1) * np.linalg.norm(a)

        v = [a[j] - alfa if (j == i) else a[j] for j in range(m)]
        vTv = np.linalg.norm(v)**2

        for k in range(n):
            vTx = 0
            for j in range(m): vTx += v[j] * A[j][k]
            for j in range(m): A[j][k] -= 2 * (vTx/vTv) * v[j]

        vTx = 0
        for j in range(m): vTx += v[j] * b[j]
        for j in range(m): b[j] -= 2 * (vTx/vTv) * v[j]

    x = sucesiva_hacia_atras(A[:n], b[:n])

    return x


# Ejecuta una función polinómica para una entrada t con parámetros x
def polinomio(n, t, x):
    ft = sum([x[i]*t**i for i in range(n)])
    return ft


# Dibuja los puntos y función resultante en el plano
def graficar(n, te, ye, tv, yv):
    """
    Entrada: un entero n y cuatro vectores te, ye, tv, yv que contienen
            los datos de entrenamiento y validación, respectivamente.
    Salida: parámetros x del ajuste de datos, tiempo de cómputo y error
            cuadrático medio, para ambos métodos.
    """

    inicio = time.time()
    x_en = ecuaciones_normales(n, te, ye)
    tiempo_en = time.time() - inicio

    inicio = time.time()
    x_hh = householder(n, te, ye)
    tiempo_hh = time.time() - inicio
    
    me, mv = len(te), len(tv)
    min_t, max_t = min(min(te), min(tv)), max(max(te), max(tv))
    t_funcion = np.linspace(min_t, max_t, 1000)
    y_funcion = [polinomio(n, i, x_en) for i in t_funcion]
    plt.plot(t_funcion, y_funcion, color="black")

    for i in range(me): plt.plot(te[i], ye[i], marker=".", color="blue")
    for i in range(mv): plt.plot(tv[i], yv[i], marker=".", color="red")
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.show()

    yp = [polinomio(n, i, x_en) for i in tv]
    ecm_en = sum([(yp[i] - yv[i])**2 for i in range(mv)]) / mv
    yp = [polinomio(n, i, x_hh) for i in tv]
    ecm_hh = sum([(yp[i] - yv[i])**2 for i in range(mv)]) / mv

    print("MÉTODO DE ECUACIONES NORMALES")
    print("x = {0}\nECM = {1}\nTiempo = {2}\n".format(x_en, ecm_en, tiempo_en))
    print("MÉTODO DE HOUSEHOLDER")
    print("x = {0}\nECM = {1}\nTiempo = {2}\n".format(x_hh, ecm_hh, tiempo_hh))

    return x_en, x_hh, tiempo_en, tiempo_hh, ecm_en, ecm_hh


# Procesamiento de los datos (adaptado a los ejemplos)
def procesar(url):
    """
    Entrada: url del conjunto de datos.
    Salida: datos de entrenamiento te (entradas) y ye (salidas), y
            de validación tv (entradas) y yv (salidas).
    """

    N = 50
    datos = pd.read_csv(url)
    borrar = ["Province/State", "Country/Region", "Lat", "Long"]
    for i in borrar: datos = datos.drop(i, axis=1)
    t = [i + 1 for i in range(N)]
    y = [i for i in datos.sum().tolist()[-N:]]

    te = [t[i] for i in range(N) if (i % 2 == 0)]
    ye = [y[i] for i in range(N) if (i % 2 == 0)]
    tv = [t[i] for i in range(N) if (i % 2 != 0)]
    yv = [y[i] for i in range(N) if (i % 2 != 0)]

    return te, ye, tv, yv


# EJEMPLOS DE PRUEBA (También se ecuentran en el informe)
def main():

    print("EJEMPLO 1: Muertos por COVID-19")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/muertos.csv"
    te, ye, tv, yv = procesar(url)
    graficar(6, te, ye, tv, yv)

    print("EJEMPLO 2: Recuperados de COVID-19")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/recuperados.csv"
    te, ye, tv, yv = procesar(url)
    graficar(5, te, ye, tv, yv)


main()


# ----------------------------------------------------------------------
# ANÁLISIS DE COMPLEJIDAD Y EXACTITUD DE LOS MÉTODOS

# Conocer los resultados de cada n y sus estadísticas
def estadisticas_n(url):
    n = [i for i in range(2, 7)]
    te, ye, tv, yv = procesar(url)
    t_en, t_hh, e_en, e_hh = [], [], [], []

    for i in range(2, 7):
        print("n =", i)
        _, _, tiempo_en, tiempo_hh, ecm_en, ecm_hh = graficar(i, te, ye, tv, yv)
        t_en += [tiempo_en]
        t_hh += [tiempo_hh]
        e_en += [ecm_en]
        e_hh += [ecm_hh]

    print("TIEMPO DE EJECUCIÓN")
    plt.plot(n, t_hh, marker="o", color="red")
    plt.plot(n, t_en, marker="o", color="blue")
    plt.xlabel("n")
    plt.ylabel("Tiempo")
    plt.grid()
    plt.show()

    print("ERROR CUADRÁTICO MEDIO")
    plt.plot(n, e_hh, marker="o", color="red")
    plt.plot(n, e_en, marker="o", color="blue")
    plt.xlabel("n")
    plt.ylabel("Error Cuadrático Medio")
    plt.grid()
    plt.show()


print("EJEMPLO 1: Muertos por COVID-19")
url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/muertos.csv"
estadisticas_n(url)

print("EJEMPLO 2: Recuperados de COVID-19")
url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/recuperados.csv"
estadisticas_n(url)