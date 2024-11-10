import serial
import numpy as np
import struct
import time
from typing import List, Tuple


class ESP32NeuralProcessor:
    def __init__(self, port='COM7', baudrate=115200, matrix_size=100):
        """
        Inicializa o processador neural ESP32.

        Args:
            port: Porta serial do ESP32
            baudrate: Taxa de transmissão
            matrix_size: Tamanho máximo da matriz suportado pelo ESP32
        """
        self.ser = serial.Serial(port, baudrate, timeout=5)
        self.MAX_SIZE = matrix_size
        time.sleep(2)  # Aguarda inicialização

    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiplica matrizes usando ESP32.
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError("Dimensões incompatíveis para multiplicação")

        if (A.shape[0] > self.MAX_SIZE or A.shape[1] > self.MAX_SIZE or
                B.shape[0] > self.MAX_SIZE or B.shape[1] > self.MAX_SIZE):
            raise ValueError(f"Matriz excede o tamanho máximo de {self.MAX_SIZE}")

        # Envia dimensões
        for dim in [A.shape[0], A.shape[1], B.shape[0], B.shape[1]]:
            self.ser.write(struct.pack('i', dim))

        # Envia matriz A
        for row in A:
            for val in row:
                self.ser.write(struct.pack('f', float(val)))

        # Envia matriz B
        for row in B:
            for val in row:
                self.ser.write(struct.pack('f', float(val)))

        # Lê status
        status = struct.unpack('i', self.ser.read(4))[0]
        if status == -1:
            raise ValueError("Erro no ESP32: Dimensões inválidas")

        # Recebe resultado
        result = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                result[i, j] = struct.unpack('f', self.ser.read(4))[0]

        return result


class ESP32NeuralNetwork:
    def __init__(self, layer_sizes: List[int], esp32: ESP32NeuralProcessor):
        """
        Implementa uma rede neural usando ESP32 como acelerador.

        Args:
            layer_sizes: Lista com o número de neurônios por camada
            esp32: Instância do processador ESP32
        """
        self.layer_sizes = layer_sizes
        self.esp32 = esp32
        self.weights = []
        self.biases = []

        # Inicializa pesos e biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            )
            self.biases.append(
                np.zeros((1, layer_sizes[i + 1]))
            )

    def relu(self, x: np.ndarray) -> np.ndarray:
        """Função de ativação ReLU"""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada da ReLU"""
        return np.where(x > 0, 1, 0)

    def forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Realiza forward pass usando ESP32 para multiplicação de matrizes.
        """
        activations = [X]
        zs = []

        for i in range(len(self.weights)):
            # Usa ESP32 para multiplicação de matrizes
            z = self.esp32.matrix_multiply(activations[-1], self.weights[i])
            z = z + self.biases[i]
            zs.append(z)

            # Última camada usa sigmoid, demais usam ReLU
            if i == len(self.weights) - 1:
                activation = 1 / (1 + np.exp(-z))  # sigmoid
            else:
                activation = self.relu(z)
            activations.append(activation)

        return activations, zs

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
              learning_rate: float):
        """
        Treina a rede neural usando mini-batch SGD.
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Embaralha os dados
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Processa em mini-batches
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                activations, zs = self.forward_pass(X_batch)

                # Backward pass
                delta = (activations[-1] - y_batch) * activations[-1] * (1 - activations[-1])
                deltas = [delta]

                # Calcula deltas para camadas ocultas
                for l in range(len(self.weights) - 1, 0, -1):
                    delta = self.esp32.matrix_multiply(delta, self.weights[l].T)
                    delta = delta * self.relu_derivative(zs[l - 1])
                    deltas.append(delta)
                deltas.reverse()

                # Atualiza pesos e biases
                for j in range(len(self.weights)):
                    weight_gradients = self.esp32.matrix_multiply(
                        activations[j].T, deltas[j]
                    )
                    bias_gradients = np.sum(deltas[j], axis=0, keepdims=True)

                    self.weights[j] -= learning_rate * weight_gradients
                    self.biases[j] -= learning_rate * bias_gradients

            # Calcula e mostra erro a cada época
            predictions, _ = self.forward_pass(X)
            error = np.mean(np.square(predictions[-1] - y))
            print(f"Época {epoch + 1}/{epochs}, Erro: {error:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições com a rede treinada"""
        return self.forward_pass(X)[0][-1]


# Exemplo de uso
if __name__ == "__main__":
    # Cria dados de exemplo (XOR)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)

    y = np.array([[1],
                  [1],
                  [1],
                  [0]], dtype=np.float32)

    try:
        # Inicializa ESP32
        esp32 = ESP32NeuralProcessor()

        # Cria rede neural: 2 entradas -> 8 hidden -> 1 saída
        nn = ESP32NeuralNetwork([2, 10, 1], esp32)

        print("Iniciando treinamento...")
        nn.train(X, y, epochs=100, batch_size=4, learning_rate=0.1)

        # Testa a rede
        predictions = nn.predict(X)
        print("\nResultados:")
        for i in range(len(X)):
            print(f"Entrada: {X[i]} -> Saída: {predictions[i][0]:.4f} (Esperado: {y[i][0]})")
            if predictions[i][0] >= 0.6055:
             print("certo")
            else:
             print("npp")
    finally:
        esp32.ser.close()

