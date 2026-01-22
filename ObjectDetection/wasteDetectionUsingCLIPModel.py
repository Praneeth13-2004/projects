import cv2
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


labels = [
    "broken mobile",
    "broken watch",
    "torch light",
    "car",
    "light",
    "fan",
    "television",
    "laptop",
    "human",
    "fitness watch",
    "medical device",
    "washing machine",
    "fridge",
    "laptop bluetooth mouse",
    "laptop keyboard","human face","book","headphones","earphones","bag","plastic cover","plastic waste","vegetable waste","pen","glass",
    "pencil","box","table","wallet","mirror","hair coomb"
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    idx = outputs.logits_per_image.argmax(dim=1).item()
    label = labels[idx]

    cv2.putText(
        frame,
        label,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    
    cv2.imshow("CLIP Object Recognition", frame)

    
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
