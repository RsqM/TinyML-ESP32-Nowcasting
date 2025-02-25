//-----Main TensorFlow Library for ESP32------
#include <TensorFlowLite_ESP32.h>

//-----------TensorFlow Helpers---------------
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "model.h" //model declaration
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

//----Sensor and Peripheral Libraries and definitions----
#include <Wire.h>
#include <Adafruit_BME280.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels
#define BME280 0X76

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
Adafruit_BME280 bme;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;
constexpr int kTensorArenaSize = 2500; //Arena Size Variable - Trial-and-Error
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace declaration end




// The name of this function is important for Arduino compatibility.
void setup() {

  //--------------------------------TENSORFLOW SETUP Code------------------------------------------
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }


  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;


  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }


  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  //-----------------------------------SENSOR SETUP CODE------------------------------------------

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  if(!bme.begin(BME280)) { // Address 0x76 for BME
    Serial.println(F("BME allocation failed"));
    for(;;);
  }



}

// The name of this function is important for Arduino compatibility.
void loop() {
  //---------------------------------Display Code---------------------------------------
  float Temp = bme.readTemperature();
  float Pres = bme.readPressure()/100.0F;
  float Hum = bme.readHumidity();
  float Alt = bme.readAltitude(1013.25);
  delay(2000);
  display.clearDisplay();
  display.setTextSize(1.25);
  display.setTextColor(WHITE);
  display.setCursor(0, 6);
  // Display static text
  display.print("Temperature: ");
  display.print(Temp);
  Serial.print("Temperature = ");
  Serial.println(Temp);
  display.print("*C");
  display.setCursor(0, 18);
  display.print("Pressure: ");
  display.print(Pres);
  Serial.print("Pressure = ");
  Serial.println(Pres);
  display.print("hPa");
  display.setCursor(0, 30);
  display.print("Humidity: ");
  display.print(Hum);
  Serial.print("Humidity = ");
  Serial.println(Hum);
  display.print("%");
  display.setCursor(0, 42);
  display.print("Altitude: ");
  display.print(Alt);
  Serial.print("Altitude = ");
  Serial.println(Alt);
  display.print("m");
  
  
  //------------------------Neural Network Invoke-------------------------
  //Passing Temperature and pressure as parameters
  float x0 = 1.8 * Temp + 32;
  float x1 = Hum;

  // Quantize the input from floating-point to integer
  int8_t x0_quantized = x0 / input->params.scale + input->params.zero_point;
  int8_t x1_quantized = x1 / input->params.scale + input->params.zero_point;

  // Place the quantized input in the model's input tensor
  input->data.int8[0] = x0_quantized;
  input->data.int8[1] = x1_quantized;


  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f, %f\n",
                         static_cast<double>(x0), static_cast<double>(x1));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  Serial.print("X value ");
  Serial.print(x0);
  Serial.print(", ");
  Serial.println(x1);
  Serial.print("y quant ");
  Serial.println(y_quantized);
  Serial.print("y ");
  Serial.println(y);

  //------------------Display Predicted-------------------------
  display.setCursor(0, 54);
  display.print("Heat Index: ");
  display.print(((y-32)*5)/9);
  Serial.print("Heat Index = ");
  Serial.println(((y-32)*5)/9);
  display.print("*C");
  display.display();
  }