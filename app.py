import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('my_model (2).h5')  # Update with your model path

# Function to preprocess frame
def preprocess_frame(frame, target_size=(128, 128)):
    frame = cv2.resize(frame, target_size)
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)
    return frame

# Define your class labels
class_labels = ['Without Mask', 'Mask']  # Update with your list of class labels

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    processed_frame = preprocess_frame(frame)
    
    # Make prediction
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions)
    
    # Verify predicted_class value
    print(f"Predicted class index: {predicted_class}")
    
    # Check if predicted_class is within valid range
    if predicted_class < len(class_labels):
        label = class_labels[predicted_class]
    else:
        label = 'Unknown'
    
    # Display the frame with label
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Real-time Detection', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
