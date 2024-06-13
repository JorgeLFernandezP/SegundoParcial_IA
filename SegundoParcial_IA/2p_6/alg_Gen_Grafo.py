import random

# Definir el grafo
grafo = {
    'A': {'B': 7, 'C': 9, 'D': 10, 'E': 20},
    'B': {'A': 7, 'C': 8, 'D': 4, 'E': 11},
    'C': {'A': 9, 'B': 8, 'D': 15, 'E': 5},
    'D': {'A': 10, 'B': 4, 'C': 15, 'E': 17},
    'E': {'A': 20, 'B': 11, 'C': 5, 'D': 17},
}

# Generar una solución aleatoria (un camino en el grafo)
def generar_camino():
    nodos = list(grafo.keys())
    random.shuffle(nodos)
    return nodos

# Evaluar un camino (calcular la distancia total)
def evaluar_camino(camino):
    distancia = 0
    for i in range(len(camino) - 1):
        distancia += grafo[camino[i]][camino[i + 1]]
    return distancia

# Selección por torneo
def seleccion_por_torneo(poblacion, k=3):
    seleccionados = random.sample(poblacion, k)
    seleccionados.sort(key=lambda x: x[1])
    return seleccionados[0][0]

# Cruce de orden (Order Crossover)
def cruce(padre1, padre2):
    size = len(padre1)
    hijo = [None] * size

    # Elegir dos puntos de cruce
    start, end = sorted(random.sample(range(size), 2))
    
    # Copiar segmento del primer padre al hijo
    hijo[start:end] = padre1[start:end]

    # Rellenar el hijo con los genes del segundo padre
    idx = end
    for gene in padre2:
        if gene not in hijo:
            if idx >= size:
                idx = 0
            hijo[idx] = gene
            idx += 1
            
    return hijo

# Mutación (intercambiar dos nodos)
def mutacion(camino):
    a, b = random.sample(range(len(camino)), 2)
    camino[a], camino[b] = camino[b], camino[a]
    return camino

# Parámetros del algoritmo genético
tamano_poblacion = 50
numero_generaciones = 10
probabilidad_cruce = 0.8
probabilidad_mutacion = 0.2

# Inicializar la población
poblacion = [(generar_camino(), None) for _ in range(tamano_poblacion)]

# Evaluar la población inicial
for i in range(tamano_poblacion):
    camino = poblacion[i][0]
    poblacion[i] = (camino, evaluar_camino(camino))

# Algoritmo genético
for generacion in range(numero_generaciones):
    nueva_poblacion = []
    
    # Crear nueva población
    for _ in range(tamano_poblacion):
        # Selección
        padre1 = seleccion_por_torneo(poblacion)
        padre2 = seleccion_por_torneo(poblacion)
        
        # Cruce
        if random.random() < probabilidad_cruce:
            hijo = cruce(padre1, padre2)
        else:
            hijo = padre1
        
        # Mutación
        if random.random() < probabilidad_mutacion:
            hijo = mutacion(hijo)
        
        # Evaluar el nuevo individuo
        nueva_poblacion.append((hijo, evaluar_camino(hijo)))
    
    # Reemplazar la antigua población con la nueva
    poblacion = nueva_poblacion
    
    # Imprimir la mejor solución de la generación actual
    mejor_camino = min(poblacion, key=lambda x: x[1])
    print(f"Generación {generacion + 1}: Mejor distancia = {mejor_camino[1]}, Camino = {mejor_camino[0]}")

# Imprimir la mejor solución final
mejor_camino = min(poblacion, key=lambda x: x[1])
print(f"Mejor distancia encontrada: {mejor_camino[1]}")
print(f"Mejor camino: {mejor_camino[0]}")
