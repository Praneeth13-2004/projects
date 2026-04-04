import cv2
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


labels = ["laptop", "mobile", "book", "person", "car"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    predicted_index = outputs.logits_per_image.argmax().item()
    predicted_label = labels[predicted_index]

    cv2.putText(frame, predicted_label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
