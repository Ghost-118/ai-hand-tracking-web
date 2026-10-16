import cv2
from cvzone.HandTrackingModule import HandDetector
from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Aumentamos la confianza al 80% para evitar falsos positivos
detector = HandDetector(detectionCon=0.8, maxHands=1)

def count_fingers_accurate(hand):
    """
    Calcula con precisión los dedos levantados comparando
    las puntas con las articulaciones intermedias (PIPs).
    """
    lmList = hand['lmList']  # Lista de 21 landmarks [x, y, z]
    handType = hand['type']  # 'Left' o 'Right'
    fingers = []

    # 1. PULGAR: Evaluamos la posición horizontal según si es mano derecha o izquierda
    # Tip: landmark 4, Joint: landmark 3 / 2
    if handType == "Right":
        if lmList[4][0] < lmList[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if lmList[4][0] > lmList[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)

    # 2. OTROS 4 DEDOS (Índice: 8, Medio: 12, Anular: 16, Meñique: 20)
    # Comparamos la punta (tip) contra la articulación (pip - 2 posiciones atrás)
    tip_ids = [8, 12, 16, 20]
    for tip in tip_ids:
        if lmList[tip][1] < lmList[tip - 2][1]:  # En OpenCV 'y' disminuye hacia arriba
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

def generate_frames():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Espejo horizontal para mayor comodidad
        frame = cv2.flip(frame, 1)

        hands, frame = detector.findHands(frame, draw=True)
        finger_count = 0

        if hands:
            hand = hands[0]
            # Usamos nuestra función personalizada con lógica exacta de coordenadas
            finger_count = count_fingers_accurate(hand)

        # Overlay del contador futurista
        cv2.rectangle(frame, (20, 20), (280, 80), (10, 10, 20), -1)
        cv2.rectangle(frame, (20, 20), (280, 80), (0, 240, 255), 1)

        text = f"DEDOS: {finger_count}"
        cv2.putText(frame, text, (35, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)