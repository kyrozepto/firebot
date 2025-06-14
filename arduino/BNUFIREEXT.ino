#include <WiFi.h>
#include <Firebase_ESP_Client.h>

#define WIFI_SSID "#"
#define WIFI_PASSWORD "#"
#define API_KEY "#"
#define DATABASE_URL "#"
#define USER_EMAIL "#"
#define USER_PASSWORD "#"

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

unsigned long lastSendTime = 0;
const unsigned long interval = 500;

// Pin sensor dan output
const int fireAnalogPin = 35;
const int fireDigitalPin = 34;
const int relayKipas = 25;

// Motor
const int motor1Pin1 = 22;
const int motor1Pin2 = 23;
const int enA = 21;
const int motor2Pin1 = 17;
const int motor2Pin2 = 18;
const int enB = 16;

int dutyCycle = 180;

// EMA Filter
float emaValue = 0;
const float alpha = 0.5;  // makin kecil makin halus

int getFilteredFireValue() {
  int raw = analogRead(fireAnalogPin);
  emaValue = alpha * raw + (1 - alpha) * emaValue;
  return (int)emaValue;
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }
  Serial.println("\nWiFi connected");

  config.api_key = API_KEY;
  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;
  config.database_url = DATABASE_URL;
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(fireAnalogPin, INPUT);
  pinMode(fireDigitalPin, INPUT);
  pinMode(relayKipas, OUTPUT);
  digitalWrite(relayKipas, LOW);

  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);

  ledcSetup(0, 1000, 8);
  ledcSetup(1, 1000, 8);
  ledcAttachPin(enA, 0);
  ledcAttachPin(enB, 1);
}

void motorStop() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
  ledcWrite(0, 0);
  ledcWrite(1, 0);
}

void motorForward() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  ledcWrite(0, dutyCycle);
  ledcWrite(1, dutyCycle);
}

void motorBackward() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
  ledcWrite(0, dutyCycle);
  ledcWrite(1, dutyCycle);
}

void motorLeft() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  ledcWrite(0, dutyCycle);
  ledcWrite(1, dutyCycle);
}

void motorRight() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
  ledcWrite(0, dutyCycle);
  ledcWrite(1, dutyCycle);
}

void loop() {
  unsigned long currentTime = millis();
  if (currentTime - lastSendTime >= interval) {
    lastSendTime = currentTime;

    // ==== BACA SENSOR API ====
    int fireValue = getFilteredFireValue();
    int fireStatus = digitalRead(fireDigitalPin);
    Firebase.RTDB.setInt(&fbdo, "/MIKROKONTROLER/FIREVALUE", fireValue);
    Firebase.RTDB.setInt(&fbdo, "/MIKROKONTROLER/FIRESTATUS", fireStatus);

    Serial.println("========== STATUS ROBOT ==========");
    Serial.println("FireStatus (digital) : " + String(fireStatus == 1 ? "API TERDETEKSI" : "TIDAK ADA API"));
    Serial.println("FireValue  (analog)  : " + String(fireValue));

    // ==== BACA MODE ====
    int mode = 0;
    if (Firebase.RTDB.get(&fbdo, "/REMOTE/AUTO")) {
      mode = fbdo.stringData().toInt();
    } else {
      Serial.println("Gagal membaca /REMOTE/AUTO: " + fbdo.errorReason());
    }
    Serial.println("Mode Kendali         : " + String(mode == 0 ? "MANUAL" : "AUTO"));

    // ======================= MODE MANUAL =======================
    if (mode == 0) {
      int cmd = 0;
      if (Firebase.RTDB.get(&fbdo, "/REMOTE/COMMAND")) {
        cmd = fbdo.stringData().toInt();
      } else {
        Serial.println("Gagal membaca /REMOTE/COMMAND: " + fbdo.errorReason());
      }

      Serial.print("Perintah Manual      : ");
      switch (cmd) {
        case 1: Serial.println("MAJU"); motorForward(); break;
        case 2: Serial.println("KIRI"); motorLeft(); break;
        case 3: Serial.println("KANAN"); motorRight(); break;
        case 4: Serial.println("MUNDUR"); motorBackward(); break;
        default: Serial.println("BERHENTI"); motorStop(); break;
      }

      int fireRelay = 0;
      if (Firebase.RTDB.get(&fbdo, "/REMOTE/FIRE")) {
        fireRelay = fbdo.stringData().toInt();
      } else {
        Serial.println("Gagal membaca /REMOTE/FIRE: " + fbdo.errorReason());
      }

      digitalWrite(relayKipas, fireRelay == 1 ? LOW : HIGH);  // Aktif LOW
      Serial.println("Relay Kipas Manual   : " + String(fireRelay == 1 ? "NYALA" : "MATI"));
    }

    // ======================= MODE OTOMATIS =======================
    else {
      int cmd = 0;
      bool commandAvailable = false;
      if (Firebase.RTDB.get(&fbdo, "/firebot/command")) {
        cmd = fbdo.stringData().toInt();
        commandAvailable = true;
      } else {
        Serial.println("Gagal membaca /firebot/command: " + fbdo.errorReason());
      }

      Serial.print("Perintah YOLO        : ");
      switch (cmd) {
        case 1: Serial.println("MAJU"); break;
        case 2: Serial.println("KIRI"); break;
        case 3: Serial.println("KANAN"); break;
        case 4: Serial.println("MUNDUR"); break;
        default: Serial.println("TIDAK ADA / STOP"); break;
      }

      // Jika tidak ada command dari YOLO atau command = 0
      if (!commandAvailable || cmd == 0) {
        Serial.println(">> Tidak ada perintah dari YOLO.");
        motorStop();
        digitalWrite(relayKipas, HIGH); // MATI
        Serial.println(">> Aksi              : MOTOR BERHENTI, RELAY MATI");
      }

      // Koreksi otomatis berdasarkan jarak api (fireValue)
      else if (fireValue < 1) {
        Serial.println(">> Api terlalu dekat (<1)");
        motorBackward();
        digitalWrite(relayKipas, LOW); // MATI
        Serial.println(">> Aksi              : MUNDUR, RELAY MATI");
      }

      else if (fireValue <= 150) {
        Serial.println(">> Jarak optimal (10–150)");
        motorStop();
        digitalWrite(relayKipas, LOW); // NYALA
        Serial.println(">> Aksi              : BERHENTI, RELAY NYALA");
      }
      else if (fireValue > 150 && fireValue < 350 ) {
        Serial.println(">> MULAI PEMADAMAN(150-500)");
        switch (cmd) {
          case 1: motorForward(); break;
          case 2: motorLeft(); break;
          case 3: motorRight(); break;
          case 4: motorBackward(); break;
          default: motorStop(); break;
        }
        digitalWrite(relayKipas, LOW); // MATI
        Serial.println(">> Aksi              : JALANKAN PERINTAH YOLO, RELAY NYALA");
      }

      else {
        // Jalankan command YOLO (jika fireValue > 300)
        Serial.println(">> Api terlalu jauh (>150)");
        switch (cmd) {
          case 1: motorForward(); break;
          case 2: motorLeft(); break;
          case 3: motorRight(); break;
          case 4: motorBackward(); break;
          default: motorStop(); break;
        }
        digitalWrite(relayKipas, HIGH); // MATI
        Serial.println(">> Aksi              : JALANKAN PERINTAH YOLO, RELAY MATI");
      }
    }

    Serial.println("====================================\n");
  }
}



// void loop() {
//   unsigned long currentTime = millis();
//   if (currentTime - lastSendTime >= interval) {
//     lastSendTime = currentTime;

//     int fireValue = getFilteredFireValue();
//     int fireStatus = digitalRead(fireDigitalPin);
//     Firebase.RTDB.setInt(&fbdo, "/MIKROKONTROLER/FIREVALUE", fireValue);
//     Firebase.RTDB.setInt(&fbdo, "/MIKROKONTROLER/FIRESTATUS", fireStatus);

//     Serial.println("========== STATUS ==========");
//     Serial.println("FireStatus (digital) : " + String(fireStatus));
//     Serial.println("FireValue  (analog)  : " + String(fireValue));

//     int mode = 0;
//     if (Firebase.RTDB.get(&fbdo, "/REMOTE/AUTO")) {
//       mode = fbdo.stringData().toInt();
//     } else {
//       Serial.println("Gagal baca /REMOTE/AUTO: " + fbdo.errorReason());
//     }

//     Serial.println("Mode: " + String(mode == 0 ? "MANUAL" : "AUTO"));

//     if (mode == 0) {
//       int cmd = 0;
//       if (Firebase.RTDB.get(&fbdo, "/REMOTE/COMMAND")) {
//         cmd = fbdo.stringData().toInt();
//       } else {
//         Serial.println("Gagal baca /REMOTE/COMMAND: " + fbdo.errorReason());
//       }

//       Serial.println("Manual Command: " + String(cmd));
//       switch (cmd) {
//         case 1: motorForward(); break;
//         case 2: motorLeft(); break;
//         case 3: motorRight(); break;
//         case 4: motorBackward(); break;
//         default: motorStop(); break;
//       }

//       int fireRelay = 0;
//       if (Firebase.RTDB.get(&fbdo, "/REMOTE/FIRE")) {
//         fireRelay = fbdo.stringData().toInt();
//       } else {
//         Serial.println("Gagal baca /REMOTE/FIRE: " + fbdo.errorReason());
//       }

//       digitalWrite(relayKipas, fireRelay == 1 ? HIGH : LOW);
//       Serial.println("Relay Kipas: " + String(fireRelay == 1 ? "NYALA" : "MATI"));

//     } else {
//       int cmd = 0;
//       bool commandAvailable = false;
//       if (Firebase.RTDB.get(&fbdo, "/firebot/command")) {
//         cmd = fbdo.stringData().toInt();
//         commandAvailable = true;
//       } else {
//         Serial.println("Gagal baca /firebot/command: " + fbdo.errorReason());
//       }

//       Serial.println("AUTO Command (YOLO): " + String(cmd));
//       Serial.println("FireValue (analog): " + String(fireValue));

//       if (!commandAvailable || fireStatus == 0 || cmd == 0) {
//         motorStop();
//         digitalWrite(relayKipas, LOW);
//         Serial.println(">> Tidak ada command aktif atau api tidak terdeteksi. Motor STOP. Relay MATI.");
//         return;
//       }

//       // Koreksi posisi terhadap api
//       if (fireValue < 10) {
//         motorBackward();
//         digitalWrite(relayKipas, LOW);
//         Serial.println(">> Api terlalu dekat. MUNDUR. Relay MATI.");
//       } else if (fireValue >= 10 && fireValue <= 300) {
//         motorStop();
//         digitalWrite(relayKipas, HIGH);
//         Serial.println(">> Jarak ideal. STOP & Relay ON.");
//       } else {
//         switch (cmd) {
//           case 1: motorForward(); break;
//           case 2: motorLeft(); break;
//           case 3: motorRight(); break;
//           case 4: motorBackward(); break;
//           default: motorStop(); break;
//         }
//         digitalWrite(relayKipas, LOW);
//         Serial.println(">> Api terlalu jauh. Jalankan perintah YOLO. Relay MATI.");
//       }
//     }

//     Serial.println("===========================\n");
//   }
// }
