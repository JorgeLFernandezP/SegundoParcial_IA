def knapsack_swap_search(values, weights, capacity, max_iter=1000):
    n = len(values)
    current_solution = [0] * n  # Start with all items not in the knapsack
    current_weight = 0
    current_value = 0
    
    # Initial solution
    best_solution = list(current_solution)
    best_value = 0
    
    # Evaluate the initial solution
    for i in range(n):
        if current_solution[i] == 1:
            current_value += values[i]
            current_weight += weights[i]
            if current_weight > capacity:
                return best_solution, best_value
    best_value = current_value
    
    # Search loop
    for _ in range(max_iter):
        # Generate a neighbor
        neighbor = list(current_solution)
        i = random.randint(0, n-1)  # Select a random item to toggle
        neighbor[i] = 1 - neighbor[i]  # Toggle the item
        
        # Evaluate the neighbor
        neighbor_value = 0
        neighbor_weight = 0
        for j in range(n):
            if neighbor[j] == 1:
                neighbor_value += values[j]
                neighbor_weight += weights[j]
                if neighbor_weight > capacity:
                    break
        
        # If the neighbor is feasible and improves the solution, accept it
        if neighbor_weight <= capacity and neighbor_value > best_value:
            best_solution = list(neighbor)
            best_value = neighbor_value
            
        current_solution = list(neighbor)
        current_value = neighbor_value
        current_weight = neighbor_weight
    
    return best_solution, best_value

# Ejemplo de uso
import random

# Datos de ejemplo
values = [3, 4, 5, 6]
weights = [2, 3, 4, 5]
capacity = 8

# Resolver el problema de la mochila
best_solution, best_value = knapsack_swap_search(values, weights, capacity)
print("Mejor solución:", best_solution)
print("Mejor valor:", best_value)
