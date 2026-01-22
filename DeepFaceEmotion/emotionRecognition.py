import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


deepface_emotions = ['angry', 'happy', 'sad','neutral']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    resized_frame = cv2.resize(frame, (1080,1080))

    try:
        result = DeepFace.analyze(
            resized_frame,
            actions=['emotion'],
            enforce_detection=False
        )

        if result:
            emotions = {
                emotion: result[0]['emotion'].get(emotion, 0)
                for emotion in deepface_emotions
            }

            dominant_emotion = max(emotions, key=emotions.get)
            cv2.putText(
                frame,
                f"Emotion: {dominant_emotion}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    except Exception as e:
        print(f"Error analyzing frame: {e}")

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
