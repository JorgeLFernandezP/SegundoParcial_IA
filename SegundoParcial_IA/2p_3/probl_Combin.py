import random

# Generar una solución inicial aleatoria
def generar_solucion_inicial(ciudades):
    solucion = list(ciudades)
    random.shuffle(solucion)
    return solucion

# Función objetivo (distancia total del recorrido)
def calcular_distancia_total(solucion, distancias):
    distancia_total = 0
    for i in range(len(solucion)):
        distancia_total += distancias[solucion[i-1]][solucion[i]]
    return distancia_total

# Generar vecinos (intercambiar dos ciudades)
def generar_vecinos(solucion):
    vecinos = []
    for i in range(len(solucion)):
        for j in range(i + 1, len(solucion)):
            vecino = solucion[:]
            vecino[i], vecino[j] = vecino[j], vecino[i]
            vecinos.append(vecino)
            print("vecino: ",vecino)
    return vecinos

# Búsqueda local
def busqueda_local(ciudades, distancias, iteraciones_max):
    solucion_actual = generar_solucion_inicial(ciudades)
    distancia_actual = calcular_distancia_total(solucion_actual, distancias)

    for _ in range(iteraciones_max):
        vecinos = generar_vecinos(solucion_actual)
        mejor_vecino = min(vecinos, key=lambda v: calcular_distancia_total(v, distancias))
        mejor_distancia = calcular_distancia_total(mejor_vecino, distancias)
        print("Mejor distancia en la búsqueda: ",mejor_distancia)

        if mejor_distancia < distancia_actual:
            solucion_actual = mejor_vecino
            distancia_actual = mejor_distancia
        else:
            break

    return solucion_actual, distancia_actual

# Ejemplo de uso
ciudades = [0, 1, 2, 3, 4]
distancias = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 15],
    [25, 30, 20, 15, 0]
]
iteraciones_max = 100

solucion, distancia = busqueda_local(ciudades, distancias, iteraciones_max)
print("Mejor solución:", solucion)
print("Distancia total:", distancia)
