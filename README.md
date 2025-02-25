# Nowcasting Model on ESP32

## Overview
This project implements a nowcasting model on the ESP32 microcontroller using TinyML. The model is deployed using TensorFlow Lite for Microcontrollers (TFLM) to perform in-situ inference. The implementation is designed to work within the constraints of the ESP32, using an arena size of 2048.

## Features
- Deploys a neural network on the ESP32
- Utilizes TensorFlow Lite for Microcontrollers (TFLM)
- Runs on an optimized memory footprint
- Performs real-time nowcasting inference directly on the microcontroller

## Included Files
- **`NeuralNetworkESP32.ino`**: The Arduino sketch for ESP32 integration
- **`model.cpp`**: The compiled model implementation
- **`model.h`**: The model header file with weights and network structure

## Requirements
### Hardware
- ESP32 Development Board
- USB Cable for flashing and debugging

### Software
- Arduino IDE with ESP32 board support
- TensorFlow Lite for Microcontrollers library
- Required dependencies installed via Arduino Library Manager

## Installation and Setup
1. Install the **Arduino IDE** and set up ESP32 board support by adding the ESP32 package.
2. Install necessary libraries:
   - TensorFlow Lite for Microcontrollers
   - Additional ESP32-specific dependencies if required
3. Open `NeuralNetworkESP32.ino` in Arduino IDE.
4. Compile and upload the sketch to the ESP32.
5. Monitor serial output for inference results.

## Notes
- The arena size is set to **2048** to balance memory usage and model performance.
- Ensure sufficient power supply for stable operation.
- Debugging can be done using the Serial Monitor in Arduino IDE.

## Disclaimer
TensorFlow Lite for Microcontrollers is still evolving and may introduce breaking changes. Performance and stability may vary across different ESP32 board versions and configurations. Testing and adjustments may be required for optimal deployment.

