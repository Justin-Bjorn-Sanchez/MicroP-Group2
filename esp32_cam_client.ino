/*
  esp32_cam_client.ino - Sends camera frames to server.py for solder joint detection

  The ESP32 takes a JPEG every CAPTURE_INTERVAL_MS, POSTs it to /predict on the
  server, and prints the result (OK / DEFECTS + counts) to the Serial Monitor.
  The web app's "ESP32 camera" tab shows the same frames with boxes.

  ARDUINO IDE SETUP
    - Board package: "esp32" by Espressif (Boards Manager), version 3.x
    - Pick your board below (BOARD_...) and in Tools > Board:
        XIAO ESP32S3 Sense -> "XIAO_ESP32S3", Tools > PSRAM: "OPI PSRAM"
        AI-Thinker ESP32-CAM -> "AI Thinker ESP32-CAM"
    - Serial Monitor at 115200 baud

  SERVER_URL
    Codespaces: Ports tab -> port 8000 -> set visibility to Public -> copy the
                forwarded address, e.g. https://name-8000.app.github.dev
    Your PC:    http://<your PC's IP on the same Wi-Fi>:8000   (find it with ipconfig)
    Keep "/predict?source=esp32" at the end.
*/

#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

// ======================= EDIT THESE =======================
const char *WIFI_SSID = "YOUR_WIFI_NAME";
const char *WIFI_PASS = "YOUR_WIFI_PASSWORD";
const char *SERVER_URL = "https://YOUR-CODESPACE-NAME-8000.app.github.dev/predict?source=esp32&conf=0.4";
const uint32_t CAPTURE_INTERVAL_MS = 1500;   // time between frames (Render free tier needs ~1 s per frame)

// Pick ONE board:
#define BOARD_XIAO_ESP32S3_SENSE
// #define BOARD_AI_THINKER_ESP32CAM
// ==========================================================

#if defined(BOARD_XIAO_ESP32S3_SENSE)
  #define PWDN_GPIO_NUM  -1
  #define RESET_GPIO_NUM -1
  #define XCLK_GPIO_NUM  10
  #define SIOD_GPIO_NUM  40
  #define SIOC_GPIO_NUM  39
  #define Y9_GPIO_NUM    48
  #define Y8_GPIO_NUM    11
  #define Y7_GPIO_NUM    12
  #define Y6_GPIO_NUM    14
  #define Y5_GPIO_NUM    16
  #define Y4_GPIO_NUM    18
  #define Y3_GPIO_NUM    17
  #define Y2_GPIO_NUM    15
  #define VSYNC_GPIO_NUM 38
  #define HREF_GPIO_NUM  47
  #define PCLK_GPIO_NUM  13
#elif defined(BOARD_AI_THINKER_ESP32CAM)
  #define PWDN_GPIO_NUM  32
  #define RESET_GPIO_NUM -1
  #define XCLK_GPIO_NUM   0
  #define SIOD_GPIO_NUM  26
  #define SIOC_GPIO_NUM  27
  #define Y9_GPIO_NUM    35
  #define Y8_GPIO_NUM    34
  #define Y7_GPIO_NUM    39
  #define Y6_GPIO_NUM    36
  #define Y5_GPIO_NUM    21
  #define Y4_GPIO_NUM    19
  #define Y3_GPIO_NUM    18
  #define Y2_GPIO_NUM     5
  #define VSYNC_GPIO_NUM 25
  #define HREF_GPIO_NUM  23
  #define PCLK_GPIO_NUM  22
#else
  #error "Select a board at the top of the sketch"
#endif

bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;   // always send the newest frame

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;     // 640x480 (matches the model input, faster upload)
    config.jpeg_quality = 10;              // lower = better quality, bigger file
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_VGA;     // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x (check board selection / PSRAM setting)\n", err);
    return false;
  }
  return true;
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWi-Fi connected, IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWi-Fi failed, will retry.");
  }
}

void sendFrame() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Capture failed");
    return;
  }

  HTTPClient http;
  WiFiClient plainClient;
  WiFiClientSecure secureClient;
  bool begun;
  if (strncmp(SERVER_URL, "https", 5) == 0) {
    secureClient.setInsecure();            // skip certificate check (fine for testing)
    begun = http.begin(secureClient, SERVER_URL);
  } else {
    begun = http.begin(plainClient, SERVER_URL);
  }
  if (!begun) {
    Serial.println("Bad SERVER_URL");
    esp_camera_fb_return(fb);
    return;
  }

  http.setTimeout(20000);
  http.addHeader("Content-Type", "image/jpeg");
  uint32_t t0 = millis();
  int code = http.POST(fb->buf, fb->len);
  size_t sentBytes = fb->len;
  esp_camera_fb_return(fb);

  if (code == 200) {
    String body = http.getString();
    // Response is JSON like {"status":"DEFECTS","defects":2,"counts":{"Bad":2,"Good":5},...}
    String status = "?";
    if (body.indexOf("\"status\":\"DEFECTS\"") >= 0) status = "DEFECTS";
    else if (body.indexOf("\"status\":\"OK\"") >= 0) status = "OK";
    else if (body.indexOf("\"status\":\"NO_JOINTS\"") >= 0) status = "NO JOINTS";

    int c0 = body.indexOf("\"counts\":");
    int c1 = body.indexOf('}', c0);
    String counts = (c0 >= 0 && c1 > c0) ? body.substring(c0 + 9, c1 + 1) : "";
    Serial.printf("[%lu ms, %u KB] %s %s\n", millis() - t0, (unsigned)(sentBytes / 1024),
                  status.c_str(), counts.c_str());
  } else if (code > 0) {
    Serial.printf("Server returned %d: %s\n", code, http.getString().c_str());
  } else {
    Serial.printf("Request failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\nSolder Joint Inspector - ESP32 camera client");
  if (!initCamera()) {
    while (true) delay(1000);
  }
  connectWiFi();
}

void loop() {
  static uint32_t last = 0;
  if (millis() - last >= CAPTURE_INTERVAL_MS) {
    last = millis();
    connectWiFi();
    if (WiFi.status() == WL_CONNECTED) sendFrame();
  }
  delay(10);
}
