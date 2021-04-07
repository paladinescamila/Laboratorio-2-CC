# UNIDAD 3: MÍNIMOS CUADRADOS
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Función auxiliar: Método de Sustitución Sucesiva hacia atrás
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


# Función auxiliar: Construye la matriz de eliminación para una columna
def matriz_de_eliminacion(A, k, g):
    """
    Entrada: una matriz cuadrada A, un entero k y un booleano g.
    Salida: matriz de eliminación de Gauss (si g es verdadero) o matriz de 
            eliminación de Gauss-Jordan (si g es falso) para la columna Ak.
    """
    n = len(A)
    M = np.identity(n)
    for i in range(k+1, n):
        M[i][k] = (-1) * A[i][k] / A[k][k]
    if (not g):
        for i in range(k):
            M[i][k] = (-1) * A[i][k] / A[k][k]
    
    return M


# Función auxiliar: Permuta una matriz y un vector dados con respecto a una fila de A
def permutar(A, b, k):
    """
    Entrada: una matriz A, un vector b y un entero k.
    Salida: una matriz A y un vector b permutados con respecto a k,
            además de un booleano que determina si el nuevo valor
            del pivote es cero.
    """
    n = len(A)
    i = k + 1
    while (i != n and A[k][k] == 0):
        P = np.identity(n)
        P[k], P[i], P[k][i], P[i][k] = 0, 0, 1, 1
        A = np.matmul(P, A)
        b = np.matmul(P, b)
        i += 1
    cero = A[k][k] == 0

    return A, b, cero


# Función auxiliar: Método de Eliminación de Gauss
def gauss(A, b):
    """
    Entrada: una matriz cuadrada A y un vector b.
    Salida: un vector x tal que Ax = b.
    """
    if (np.linalg.det(A) == 0):
        print("A es una matriz singular, el sistema no tiene solución.")
        return []
    else:
        n = len(A)
        for k in range(n - 1):
            if (A[k][k] == 0):
                A, b, cero = permutar(A, b, k)
                if (cero):
                    print("El sistema no tiene solución.")
                    return []
            M = matriz_de_eliminacion(A, k, 1)
            A = np.matmul(M, A)
            b = np.matmul(M, b)
        x = sucesiva_hacia_atras(A, b)

        return x


# Método de Ecuaciones Normales
def ecuaciones_normales(n, t, y):
    """
    Entrada: un entero n, un vector t y un vector y.
    Salida: un vector x de parámetros para un ajuste polinomial (orden n-1) 
            usando los datos de t (entrada) & y (salida).
    """

    A = [[i**j for j in range(n)] for i in t]
    b = [i for i in y]

    AT = np.transpose(A)
    A = np.matmul(AT, A)
    b = np.matmul(AT, b)
    x = gauss(A, b)

    return x


# Método de Transformaciones Householder
def householder(n, t, y):
    """
    Entrada: un entero n, un vector t y un vector y.
    Salida: un vector x de parámetros para un ajuste polinomial (orden n-1) 
            usando los datos de t (entrada) & y (salida).
    """

    A = [[i**j for j in range(n)] for i in t]
    b = [i for i in y]
    m = len(t)

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
            Normales y es 2 si se calcula para Transformaciones Householder, 
            y un booleano "mostrar" que si es verdadero grafica las funciones 
            e imprime los resultados.
    Salida: parámetros x del ajuste de datos, error promedio, desviación
            estándar del error, y tiempo de cómputo para el método seleccionado.
    """

    if (metodo == 1):
        if (mostrar): plt.title("Método de Ecuaciones Normales")
        inicio = time.time()
        x = ecuaciones_normales(n, te, ye)
        tiempo = time.time() - inicio
    else:
        if (mostrar): plt.title("Método de Transformaciones Householder")
        inicio = time.time()
        x = householder(n, te, ye)
        tiempo = time.time() - inicio
    
    me, mv = len(te), len(tv)
    errores = [np.abs(yv[i] - polinomio(n, tv[i], x)) for i in range(mv)]
    error_promedio, error_desviacion = np.mean(errores), np.std(errores)

    if (mostrar):

        min_t, max_t = min(min(te), min(tv)), max(max(te), max(tv))
        t_funcion = np.linspace(min_t, max_t, 1000)
        y_funcion = [polinomio(n, i, x) for i in t_funcion]
        plt.plot(t_funcion, y_funcion, color="black")

        for i in range(me): plt.plot(te[i], ye[i], marker="o", markersize=4, color="blue")
        for i in range(mv): plt.plot(tv[i], yv[i], marker="o", markersize=4, color="red")

        plt.xlabel('t')
        plt.ylabel('y')
        plt.grid()
        plt.show()

        print("x = {0}".format(x))
        print("Error (promedio) = {0:.2f}".format(error_promedio))
        print("Error (desviación estándar) = {0:.2f}".format(error_desviacion))
        print("Tiempo = {0:.5f}s\n".format(tiempo))

    return x, error_promedio, error_desviacion, tiempo


# Procesamiento de los datos (adaptado a los ejemplos)
def procesar(url):
    """
    Entrada: url del conjunto de datos.
    Salida: datos de entrenamiento te (entradas) y ye (salidas), y
            de validación tv (entradas) y yv (salidas).
    """

    N = 50
    datos = pd.read_csv(url, header=None)
    t = datos[0].tolist()[-N:]
    y = datos[1].tolist()[-N:]

    te = [t[i] for i in range(N) if (i % 2 == 0)]
    ye = [y[i] for i in range(N) if (i % 2 == 0)]
    tv = [t[i] for i in range(N) if (i % 2 != 0)]
    yv = [y[i] for i in range(N) if (i % 2 != 0)]

    return te, ye, tv, yv


# EJEMPLOS DE PRUEBA (También se encuentran en el informe)
def main():

    print("EJEMPLO 1")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/muertos.csv"
    # url = "muertos.csv" # Usar esta URL en caso de no tener conexión a internet
    te, ye, tv, yv = procesar(url)
    resolver(5, te, ye, tv, yv, 1, True)
    resolver(5, te, ye, tv, yv, 2, True)
    # x2, _, _, _ = resolver(5, te, ye, tv, yv, 1, True)
    # x2, _, _, _ = resolver(5, te, ye, tv, yv, 2, True)
    # for i in range(5): print("\t\t$x_{0}$ & {1} & {2} \\\\ \hline".format(i+1, x1[i], x2[i]))

    print("EJEMPLO 2")
    url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/recuperados.csv"
    # url = "recuperados.csv" # Usar esta URL en caso de no tener conexión a internet
    te, ye, tv, yv = procesar(url)
    resolver(8, te, ye, tv, yv, 1, True)
    resolver(8, te, ye, tv, yv, 2, True)
    # x1, _, _, _ = resolver(8, te, ye, tv, yv, 1, True)
    # x2, _, _, _ = resolver(8, te, ye, tv, yv, 2, True)
    # for i in range(8): print("\t\t$x_{0}$ & {1} & {2} \\\\ \hline".format(i+1, x1[i], x2[i]))


main()


# ----------------------------------------------------------------------
# ANÁLISIS DE COMPLEJIDAD Y EXACTITUD DE LOS MÉTODOS

# Comparar los métodos con los diferentes valores de n
def estadisticas(url):
    n = [i for i in range(2, 13)]
    te, ye, tv, yv = procesar(url)
    t_en, t_hh, e_en, e_hh, d_en, d_hh = [], [], [], [], [], []
    best_en, best_error_en, best_hh, best_error_hh = 13, float("inf"), 13, float("inf")

    for i in n:

        x, prom_en, desv_en, tiempo_en = resolver(i, te, ye, tv, yv, 1, False)
        x, prom_hh, desv_hh, tiempo_hh = resolver(i, te, ye, tv, yv, 2, False)
        t_en += [tiempo_en]
        t_hh += [tiempo_hh]
        e_en += [prom_en]
        e_hh += [prom_hh]
        d_en += [desv_en]
        d_hh += [desv_hh]

        if (prom_en < best_error_en): best_en, best_error_en = i, prom_en
        if (prom_hh < best_error_hh): best_hh, best_error_hh = i, prom_hh

    print("------------------------------------------------------------")
    print("                    Tiempo de ejecución                     ")
    print("------------------------------------------------------------")
    print("n\tEcuaciones Normales\tTransformaciones Householder")
    print("------------------------------------------------------------")
    for i in n: print("{0}\t{1}\t{2}".format(i, t_en[i-2], t_hh[i-2]))
    # for i in n: print("\t\t{0} & {1:.5f} & {2:.5f} \\\\ \hline".format(i, t_en[i-2], t_hh[i-2]))
    print("------------------------------------------------------------")
    plt.plot(n, t_en, marker="o", color="red")
    plt.plot(n, t_hh, marker="o", color="blue")
    plt.xlabel("n")
    plt.ylabel("Tiempo")
    plt.grid()
    plt.show()

    print("-----------------------------------------------------------------------")
    print("                                 Error                                 ")
    print("-----------------------------------------------------------------------")
    print("n\tPromedio (EN)\tPromedio (TH)\tDesviación (EN)\tDesviación (TH)")
    print("-----------------------------------------------------------------------")
    for i in n: print("{0}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.5f}".format(i, e_en[i-2], d_en[i-2], e_hh[i-2], d_hh[i-2]))
    # for i in n: print("\t\t{0} & {1:.5f} & {2:.5f} & {3:.5f} & {4:.5f} \\\\ \hline".format(i, e_en[i-2], d_en[i-2], e_hh[i-2], d_hh[i-2]))
    print("-----------------------------------------------------------------------")
    plt.plot(n, e_en, marker="o", color="red")
    plt.plot(n, e_hh, marker="o", color="blue")
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.grid()
    plt.show()

    print("Mejor n (Ecuaciones Normales) = {0}".format(best_en))
    print("Mejor n (Transformaciones Householder) = {0}\n".format(best_hh))


print("EJEMPLO 1")
url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/muertos.csv"
# url = "muertos.csv" # Usar esta URL en caso de no tener conexión a internet
estadisticas(url)

print("EJEMPLO 2")
url = "https://raw.githubusercontent.com/paladinescamila/Laboratorio-2-CC/main/recuperados.csv"
# url = "recuperados.csv" # Usar esta URL en caso de no tener conexión a internet
estadisticas(url)