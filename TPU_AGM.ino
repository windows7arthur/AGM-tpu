#include <Arduino.h>

// Tamanho máximo das matrizes
#define MAX_SIZE 100

// Buffer para armazenar as matrizes
float matrixA[MAX_SIZE][MAX_SIZE];
float matrixB[MAX_SIZE][MAX_SIZE];
float matrixResult[MAX_SIZE][MAX_SIZE];

// Dimensões das matrizes
int rowsA, colsA, rowsB, colsB;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Aguarda conexão serial
  }
}

// Função para multiplicar matrizes
void multiplyMatrices() {
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++) {
      matrixResult[i][j] = 0;
      for (int k = 0; k < colsA; k++) {
        matrixResult[i][j] += matrixA[i][k] * matrixB[k][j];
      }
    }
  }
}

// Função para receber matriz via serial
void receiveMatrix(float matrix[][MAX_SIZE], int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      while (Serial.available() < sizeof(float)) {
        delay(1);
      }
      Serial.readBytes((char*)&matrix[i][j], sizeof(float));
    }
  }
}

// Função para enviar matriz via serial
void sendMatrix(float matrix[][MAX_SIZE], int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      Serial.write((char*)&matrix[i][j], sizeof(float));
    }
  }
}

void loop() {
  if (Serial.available() >= 4) {  // Esperando 4 bytes para dimensões
    // Recebe dimensões das matrizes
    Serial.readBytes((char*)&rowsA, sizeof(int));
    Serial.readBytes((char*)&colsA, sizeof(int));
    Serial.readBytes((char*)&rowsB, sizeof(int));
    Serial.readBytes((char*)&colsB, sizeof(int));
    
    // Verifica se as dimensões são válidas
    if (colsA != rowsB || 
        rowsA > MAX_SIZE || colsA > MAX_SIZE || 
        rowsB > MAX_SIZE || colsB > MAX_SIZE) {
      // Envia código de erro
      int error = -1;
      Serial.write((char*)&error, sizeof(int));
      return;
    }
    
    // Recebe as matrizes
    receiveMatrix(matrixA, rowsA, colsA);
    receiveMatrix(matrixB, rowsB, colsB);
    
    // Realiza a multiplicação
    multiplyMatrices();
    
    // Envia código de sucesso
    int success = 1;
    Serial.write((char*)&success, sizeof(int));
    
    // Envia a matriz resultado
    sendMatrix(matrixResult, rowsA, colsB);
  }
}
