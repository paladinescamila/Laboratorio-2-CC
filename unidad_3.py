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

    A = [[i**j for j in range(n)] for i in t]

    AT = np.transpose(A)
    A = np.matmul(AT, A)
    L = np.linalg.cholesky(A)
    LT = np.transpose(L)

    ye = sucesiva_hacia_adelante(L, np.matmul(AT, y))
    x = sucesiva_hacia_atras(LT, ye)

    return x


# Método de Transformaciones Householder
def householder(n, t, y):
    """
    Entrada: un entero n, un vector t y un vector y.
    Salida: un vector x de parámetros para un ajuste polinomial (orden n-1) 
            usando los datos de t (entrada) & y (salida).
    """

    m = len(t)
    A = [[i**j for j in range(n)] for i in t]
    b = [i for i in y]

    for i in range(n):

        a = [0 for _ in range(i)] + [A[j][i] for j in range(i, m)]
        alfa = np.linalg.norm(a) if (A[i][i] < 0) else (-1) * np.linalg.norm(a)

        v = [a[j] - alfa if (j == i) else a[j] for j in range(m)]
        vTv = np.linalg.norm(v)**2

        for k in range(n):
            vTx = sum([v[j] * A[j][k] for j in range(m)])
            for j in range(m): A[j][k] -= 2 * (vTx/vTv) * v[j]

        vTx = sum([v[j] * b[j] for j in range(m)])
        for j in range(m): b[j] -= 2 * (vTx/vTv) * v[j]

    x = sucesiva_hacia_atras(A[:n], b[:n])

    return x


# Ejecuta una función polinómica para una entrada t con parámetros x
def polinomio(n, t, x):
    ft = sum([x[i]*t**i for i in range(n)])
    return ft


# Grafica los puntos y función resultante en el plano
def resolver(n, te, ye, tv, yv, metodo, mostrar):
    """
    Entrada: un entero n, cuatro vectores te, ye, tv, yv que contienen
            los datos de entrenamiento y validación, respectivamente, 
            un entero "metodo" que es 1 si se calcula para Ecuaciones
            Normales y es 2 si se calcula para Householder, y un booleano
            "mostrar" que si es verdadero grafica las funciones e imprime 
            los resultados.
    Salida: parámetros x del ajuste de datos, error cuadrático medio, y
            tiempo de cómputo para el método seleccionado.
    """

    if (metodo == 1):
        if (mostrar): print("Método de Ecuaciones Normales")
        inicio = time.time()
        x = ecuaciones_normales(n, te, ye)
        tiempo = time.time() - inicio
    else:
        if (mostrar): print("Método de Transformaciones Householder")
        inicio = time.time()
        x = householder(n, te, ye)
        tiempo = time.time() - inicio
    
    me, mv = len(te), len(tv)
    yp = [polinomio(n, i, x) for i in tv]
    ecm = sum([(yp[i] - yv[i])**2 for i in range(mv)]) / mv

    if (mostrar):
        print("x = {0}\nECM = {1}\nTiempo = {2}".format(x, ecm, tiempo))
        
        min_t, max_t = min(min(te), min(tv)), max(max(te), max(tv))
        t_funcion = np.linspace(min_t, max_t, 1000)
        y_funcion = [polinomio(n, i, x) for i in t_funcion]
        plt.plot(t_funcion, y_funcion, color="black")

        for i in range(me): plt.plot(te[i], ye[i], marker=".", color="blue")
        for i in range(mv): plt.plot(tv[i], yv[i], marker=".", color="red")
        plt.xlabel('t')
        plt.ylabel('y')
        plt.grid()
        plt.show()

    return x, ecm, tiempo


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


# EJEMPLOS DE PRUEBA (También se encuentran en el informe)
def main():

    print("EJEMPLO 1")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/muertos.csv"
    te, ye, tv, yv = procesar(url)
    resolver(6, te, ye, tv, yv, 1, True)
    resolver(6, te, ye, tv, yv, 2, True)

    print("EJEMPLO 2")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/recuperados.csv"
    te, ye, tv, yv = procesar(url)
    resolver(5, te, ye, tv, yv, 1, True)
    resolver(5, te, ye, tv, yv, 2, True)


main()


# ----------------------------------------------------------------------
# ANÁLISIS DE COMPLEJIDAD Y EXACTITUD DE LOS MÉTODOS

# Comparar los métodos con los diferentes valores de n
def estadisticas(url):
    n = [i for i in range(2, 7)]
    te, ye, tv, yv = procesar(url)
    t_en, t_hh, e_en, e_hh = [], [], [], []

    for i in n:
        x, ecm_en, tiempo_en = resolver(i, te, ye, tv, yv, 1, False)
        x, ecm_hh, tiempo_hh = resolver(i, te, ye, tv, yv, 2, False)
        t_en += [tiempo_en]
        t_hh += [tiempo_hh]
        e_en += [ecm_en]
        e_hh += [ecm_hh]

    print("------------------------------------------------------------")
    print("                    Tiempo de ejecución                     ")
    print("------------------------------------------------------------")
    print("n\tEcuaciones Normales\tTransformaciones Householder")
    print("------------------------------------------------------------")
    for i in n: print("{0}\t{1}\t{2}".format(i, t_en[i-2], t_hh[i-2]))
    print("------------------------------------------------------------")
    plt.plot(n, t_en, marker="o", color="red")
    plt.plot(n, t_hh, marker="o", color="blue")
    plt.xlabel("n")
    plt.ylabel("Tiempo")
    plt.grid()
    plt.show()

    print("------------------------------------------------------------")
    print("                   Error Cuadrático Medio                   ")
    print("------------------------------------------------------------")
    print("n\tEcuaciones Normales\tTransformaciones Householder")
    print("------------------------------------------------------------")
    for i in n: print("{0}\t{1}\t{2}".format(i, e_en[i-2], e_hh[i-2]))
    print("------------------------------------------------------------")
    plt.plot(n, e_en, marker="o", color="red")
    plt.plot(n, e_hh, marker="o", color="blue")
    plt.xlabel("n")
    plt.ylabel("Error Cuadrático Medio")
    plt.grid()
    plt.show()


print("EJEMPLO 1")
url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/muertos.csv"
estadisticas(url)

print("EJEMPLO 2")
url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/recuperados.csv"
estadisticas(url)