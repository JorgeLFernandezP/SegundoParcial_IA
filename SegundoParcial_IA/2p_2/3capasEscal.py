import numpy as np

# Carga del dataset Iris (manual)
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

# Función de activación escalón
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Normalización de los datos
def normalize(X):
    return X / X.max(axis=0)

# Codificación One-Hot
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Inicialización de pesos
def initialize_weights(input_size, hidden_size1, hidden_size2, output_size):
    wh1 = np.random.uniform(-1, 1, (input_size, hidden_size1))
    wh2 = np.random.uniform(-1, 1, (hidden_size1, hidden_size2))
    wo = np.random.uniform(-1, 1, (hidden_size2, output_size))
    return wh1, wh2, wo

# Propagación hacia adelante
def forward_propagation(X, wh1, wh2, wo):
    hidden_input1 = np.dot(X, wh1)
    hidden_output1 = step_function(hidden_input1)
    
    hidden_input2 = np.dot(hidden_output1, wh2)
    hidden_output2 = step_function(hidden_input2)
    
    output_input = np.dot(hidden_output2, wo)
    output_output = step_function(output_input)
    return hidden_output1, hidden_output2, output_output

# Backpropagation
def backward_propagation(X, y, hidden_output1, hidden_output2, output_output, wh1, wh2, wo, learning_rate):
    output_error = y - output_output
    output_delta = output_error  # Derivada de la función escalón es 1 donde hay error
    
    hidden_error2 = output_delta.dot(wo.T)
    hidden_delta2 = hidden_error2  # Derivada de la función escalón es 1 donde hay error
    
    hidden_error1 = hidden_delta2.dot(wh2.T)
    hidden_delta1 = hidden_error1  # Derivada de la función escalón es 1 donde hay error
    
    wo += hidden_output2.T.dot(output_delta) * learning_rate
    wh2 += hidden_output1.T.dot(hidden_delta2) * learning_rate
    wh1 += X.T.dot(hidden_delta1) * learning_rate
    
    return wh1, wh2, wo

# Entrenamiento de la red
def train(X, y, hidden_size1, hidden_size2, epochs, learning_rate):
    input_size = X.shape[1]
    output_size = y.shape[1]
    wh1, wh2, wo = initialize_weights(input_size, hidden_size1, hidden_size2, output_size)
    
    for epoch in range(epochs):
        hidden_output1, hidden_output2, output_output = forward_propagation(X, wh1, wh2, wo)
        wh1, wh2, wo = backward_propagation(X, y, hidden_output1, hidden_output2, output_output, wh1, wh2, wo, learning_rate)
        
        if epoch % 100 == 0:
            loss = np.mean(np.square(y - output_output))
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return wh1, wh2, wo

# Evaluación de la red
def predict(X, wh1, wh2, wo):
    _, _, output_output = forward_propagation(X, wh1, wh2, wo)
    return np.argmax(output_output, axis=1)

# Main
if __name__ == "__main__":
    X, y = load_iris()
    X = normalize(X)
    y = one_hot_encode(y, 3)
    
    hidden_size1 = 5
    hidden_size2 = 5
    epochs = 1000
    learning_rate = 0.2
    
    wh1, wh2, wo = train(X, y, hidden_size1, hidden_size2, epochs, learning_rate)
    predictions = predict(X, wh1, wh2, wo)
    accuracy = np.mean(np.argmax(y, axis=1) == predictions)
    
    print(f"Accuracy: {accuracy * 100}%")
