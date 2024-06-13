import numpy as np

# Función Sigmoid y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Codificación one-hot de las etiquetas
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

# Normalización de las características
def normalize(X):
    return X / X.max(axis=0)

# Cargar el dataset Iris manualmente
def load_iris():
    data = np.array([
        [5.1, 3.5, 1.4, 0.2, 0],
        [4.9, 3.0, 1.4, 0.2, 0],
        [4.7, 3.2, 1.3, 0.2, 0],
        [4.6, 3.1, 1.5, 0.2, 0],
        [5.0, 3.6, 1.4, 0.2, 0],
        [5.4, 3.9, 1.7, 0.4, 0],
        [4.6, 3.4, 1.4, 0.3, 0],
        [5.0, 3.4, 1.5, 0.2, 0],
        [4.4, 2.9, 1.4, 0.2, 0],
        [4.9, 3.1, 1.5, 0.1, 0],
        [5.4, 3.7, 1.5, 0.2, 0],
        [4.8, 3.4, 1.6, 0.2, 0],
        [4.8, 3.0, 1.4, 0.1, 0],
        [4.3, 3.0, 1.1, 0.1, 0],
        [5.8, 4.0, 1.2, 0.2, 0],
        [5.7, 4.4, 1.5, 0.4, 0],
        [5.4, 3.9, 1.3, 0.4, 0],
        [5.1, 3.5, 1.4, 0.3, 0],
        [5.7, 3.8, 1.7, 0.3, 0],
        [5.1, 3.8, 1.5, 0.3, 0],
        [5.4, 3.4, 1.7, 0.2, 0],
        [5.1, 3.7, 1.5, 0.4, 0],
        [4.6, 3.6, 1.0, 0.2, 0],
        [5.1, 3.3, 1.7, 0.5, 0],
        [4.8, 3.4, 1.9, 0.2, 0],
        [5.0, 3.0, 1.6, 0.2, 0],
        [5.0, 3.4, 1.6, 0.4, 0],
        [5.2, 3.5, 1.5, 0.2, 0],
        [5.2, 3.4, 1.4, 0.2, 0],
        [4.7, 3.2, 1.6, 0.2, 0],
        [4.8, 3.1, 1.6, 0.2, 0],
        [5.4, 3.4, 1.5, 0.4, 0],
        [5.2, 4.1, 1.5, 0.1, 0],
        [5.5, 4.2, 1.4, 0.2, 0],
        [4.9, 3.1, 1.5, 0.2, 0],
        [5.0, 3.2, 1.2, 0.2, 0],
        [5.5, 3.5, 1.3, 0.2, 0],
        [4.9, 3.6, 1.4, 0.1, 0],
        [4.4, 3.0, 1.3, 0.2, 0],
        [5.1, 3.4, 1.5, 0.2, 0],
        [5.0, 3.5, 1.3, 0.3, 0],
        [4.5, 2.3, 1.3, 0.3, 0],
        [4.4, 3.2, 1.3, 0.2, 0],
        [5.0, 3.5, 1.6, 0.6, 0],
        [5.1, 3.8, 1.9, 0.4, 0],
        [4.8, 3.0, 1.4, 0.3, 0],
        [5.1, 3.8, 1.6, 0.2, 0],
        [4.6, 3.2, 1.4, 0.2, 0],
        [5.3, 3.7, 1.5, 0.2, 0],
        [5.0, 3.3, 1.4, 0.2, 0],
        [7.0, 3.2, 4.7, 1.4, 1],
        [6.4, 3.2, 4.5, 1.5, 1],
        [6.9, 3.1, 4.9, 1.5, 1],
        [5.5, 2.3, 4.0, 1.3, 1],
        [6.5, 2.8, 4.6, 1.5, 1],
        [5.7, 2.8, 4.5, 1.3, 1],
        [6.3, 3.3, 4.7, 1.6, 1],
        [4.9, 2.4, 3.3, 1.0, 1],
        [6.6, 2.9, 4.6, 1.3, 1],
        [5.2, 2.7, 3.9, 1.4, 1],
        [5.0, 2.0, 3.5, 1.0, 1],
        [5.9, 3.0, 4.2, 1.5, 1],
        [6.0, 2.2, 4.0, 1.0, 1],
        [6.1, 2.9, 4.7, 1.4, 1],
        [5.6, 2.9, 3.6, 1.3, 1],
        [6.7, 3.1, 4.4, 1.4, 1],
        [5.6, 3.0, 4.5, 1.5, 1],
        [5.8, 2.7, 4.1, 1.0, 1],
        [6.2, 2.2, 4.5, 1.5, 1],
        [5.6, 2.5, 3.9, 1.1, 1],
        [5.9, 3.2, 4.8, 1.8, 1],
        [6.1, 2.8, 4.0, 1.3, 1],
        [6.3, 2.5, 4.9, 1.5, 1],
        [6.1, 2.8, 4.7, 1.2, 1],
        [6.4, 2.9, 4.3, 1.3, 1],
        [6.6, 3.0, 4.4, 1.4, 1],
        [6.8, 2.8, 4.8, 1.4, 1],
        [6.7, 3.0, 5.0, 1.7, 1],
        [6.0, 2.9, 4.5, 1.5, 1],
        [5.7, 2.6, 3.5, 1.0, 1],
        [5.5, 2.4, 3.8, 1.1, 1],
        [5.5, 2.4, 3.7, 1.0, 1],
        [5.8, 2.7, 3.9, 1.2, 1],
        [6.0, 2.7, 5.1, 1.6, 1],
        [5.4, 3.0, 4.5, 1.5, 1],
        [6.0, 3.4, 4.5, 1.6, 1],
        [6.7, 3.1, 4.7, 1.5, 1],
        [6.3, 2.3, 4.4, 1.3, 1],
        [5.6, 3.0, 4.1, 1.3, 1],
        [5.5, 2.5, 4.0, 1.3, 1],
        [5.5, 2.6, 4.4, 1.2, 1],
        [6.1, 3.0, 4.6, 1.4, 1],
        [5.8, 2.6, 4.0, 1.2, 1],
        [5.0, 2.3, 3.3, 1.0, 1],
        [5.6, 2.7, 4.2, 1.3, 1],
        [5.7, 3.0, 4.2, 1.2, 1],
        [5.7, 2.9, 4.2, 1.3, 1],
        [6.2, 2.9, 4.3, 1.3, 1],
        [5.1, 2.5, 3.0, 1.1, 1],
        [5.7, 2.8, 4.1, 1.3, 1],
        [6.3, 3.3, 6.0, 2.5, 2],
        [5.8, 2.7, 5.1, 1.9, 2],
        [7.1, 3.0, 5.9, 2.1, 2],
        [6.3, 2.9, 5.6, 1.8, 2],
        [6.5, 3.0, 5.8, 2.2, 2],
        [7.6, 3.0, 6.6, 2.1, 2],
        [4.9, 2.5, 4.5, 1.7, 2],
        [7.3, 2.9, 6.3, 1.8, 2],
        [6.7, 2.5, 5.8, 1.8, 2],
        [7.2, 3.6, 6.1, 2.5, 2],
        [6.5, 3.2, 5.1, 2.0, 2],
        [6.4, 2.7, 5.3, 1.9, 2],
        [6.8, 3.0, 5.5, 2.1, 2],
        [5.7, 2.5, 5.0, 2.0, 2],
        [5.8, 2.8, 5.1, 2.4, 2],
        [6.4, 3.2, 5.3, 2.3, 2],
        [6.5, 3.0, 5.5, 1.8, 2],
        [7.7, 3.8, 6.7, 2.2, 2],
        [7.7, 2.6, 6.9, 2.3, 2],
        [6.0, 2.2, 5.0, 1.5, 2],
        [6.9, 3.2, 5.7, 2.3, 2],
        [5.6, 2.8, 4.9, 2.0, 2],
        [7.7, 2.8, 6.7, 2.0, 2],
        [6.3, 2.7, 4.9, 1.8, 2],
        [6.7, 3.3, 5.7, 2.1, 2],
        [7.2, 3.2, 6.0, 1.8, 2],
        [6.2, 2.8, 4.8, 1.8, 2],
        [6.1, 3.0, 4.9, 1.8, 2],
        [6.4, 2.8, 5.6, 2.1, 2],
        [7.2, 3.0, 5.8, 1.6, 2],
        [7.4, 2.8, 6.1, 1.9, 2],
        [7.9, 3.8, 6.4, 2.0, 2],
        [6.4, 2.8, 5.6, 2.2, 2],
        [6.3, 2.8, 5.1, 1.5, 2],
        [6.1, 2.6, 5.6, 1.4, 2],
        [7.7, 3.0, 6.1, 2.3, 2],
        [6.3, 3.4, 5.6, 2.4, 2],
        [6.4, 3.1, 5.5, 1.8, 2],
        [6.0, 3.0, 4.8, 1.8, 2],
        [6.9, 3.1, 5.4, 2.1, 2],
        [6.7, 3.1, 5.6, 2.4, 2],
        [6.9, 3.1, 5.1, 2.3, 2],
        [5.8, 2.7, 5.1, 1.9, 2],
        [6.8, 3.2, 5.9, 2.3, 2],
        [6.7, 3.3, 5.7, 2.5, 2],
        [6.7, 3.0, 5.2, 2.3, 2],
        [6.3, 2.5, 5.0, 1.9, 2],
        [6.5, 3.0, 5.2, 2.0, 2],
        [6.2, 3.4, 5.4, 2.3, 2],
        [5.9, 3.0, 5.1, 1.8, 2]
    ])
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

# Inicialización de pesos
def initialize_weights(input_size, hidden_size, output_size):
    wh = np.random.uniform(-1, 1, (input_size, hidden_size))
    wo = np.random.uniform(-1, 1, (hidden_size, output_size))
    return wh, wo

# Propagación hacia adelante
def forward_propagation(X, wh, wo):
    hidden_input = np.dot(X, wh)
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, wo)
    output_output = sigmoid(output_input)
    return hidden_output, output_output

# Backpropagation
def backward_propagation(X, y, hidden_output, output_output, wh, wo, learning_rate):
    output_error = y - output_output
    output_delta = output_error * sigmoid_derivative(output_output)
    
    hidden_error = output_delta.dot(wo.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    
    wo += hidden_output.T.dot(output_delta) * learning_rate
    wh += X.T.dot(hidden_delta) * learning_rate
    
    return wh, wo

# Entrenamiento de la red
def train(X, y, hidden_size, epochs, learning_rate):
    input_size = X.shape[1]
    output_size = y.shape[1]
    wh, wo = initialize_weights(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        hidden_output, output_output = forward_propagation(X, wh, wo)
        wh, wo = backward_propagation(X, y, hidden_output, output_output, wh, wo, learning_rate)
        
        if epoch % 10 == 0:
            loss = np.mean(np.square(y - output_output))
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return wh, wo

# Evaluación de la red
def predict(X, wh, wo):
    _, output_output = forward_propagation(X, wh, wo)
    return np.argmax(output_output, axis=1)

# Main
if __name__ == "__main__":
    X, y = load_iris()
    X = normalize(X)
    y = one_hot_encode(y, 3)
    
    hidden_size = 5
    epochs = 100
    learning_rate = 0.4
    
    wh, wo = train(X, y, hidden_size, epochs, learning_rate)
    predictions = predict(X, wh, wo)
    accuracy = np.mean(np.argmax(y, axis=1) == predictions)
    
    print(f"Accuracy: {accuracy * 100}%")
