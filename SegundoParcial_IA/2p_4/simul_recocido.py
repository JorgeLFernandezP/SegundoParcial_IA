import random
import math

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

# Generar un vecino (intercambiar dos ciudades)
def generar_vecino(solucion):
    vecino = solucion[:]
    i, j = random.sample(range(len(solucion)), 2)
    vecino[i], vecino[j] = vecino[j], vecino[i]
    return vecino

# Recocido Simulado
def recocido_simulado(ciudades, distancias, temp_inicial, temp_min, alfa, iteraciones_max):
    solucion_actual = generar_solucion_inicial(ciudades)
    distancia_actual = calcular_distancia_total(solucion_actual, distancias)
    mejor_solucion = solucion_actual[:]
    mejor_distancia = distancia_actual

    temperatura = temp_inicial
    

    while temperatura > temp_min:
        for _ in range(iteraciones_max):
            vecino = generar_vecino(solucion_actual)
            distancia_vecino = calcular_distancia_total(vecino, distancias)
            delta = distancia_vecino - distancia_actual

            if delta < 0 or random.random() < math.exp(-delta / temperatura):
                solucion_actual = vecino
                distancia_actual = distancia_vecino

                if distancia_vecino < mejor_distancia:
                    mejor_solucion = vecino
                    mejor_distancia = distancia_vecino

        temperatura *= alfa
        print("Temperatura: ", temperatura)

    return mejor_solucion, mejor_distancia

# Ejemplo de uso
ciudades = [0, 1, 2, 3, 4]
distancias = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 15],
    [25, 30, 20, 15, 0]
]
temp_inicial = 10
temp_min = 1
alfa = 0.995
iteraciones_max = 10

solucion, distancia = recocido_simulado(ciudades, distancias, temp_inicial, temp_min, alfa, iteraciones_max)
print("Mejor solución:", solucion)
print("Distancia total:", distancia)
