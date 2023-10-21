import cv2
import numpy as np
import sys

def load_yolo_model():
    # Load YOLO model using cv2.dnn module
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_id_card(image, net, classes, output_layers):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust the confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Apply Non-Maximum Suppression

    cropped_id_card = None
    for i in indices:
        #i = i[0]
        print(class_ids)
        if classes[class_ids[i]] == 'person':  # Change 'person' to the class label for ID card in your dataset
            x, y, w, h = boxes[i]
            cropped_id_card = image[y:y+h, x:x+w]

    return cropped_id_card

# Main function
if __name__ == "__main__":
    image_path = sys.argv[1]
    net, classes, output_layers = load_yolo_model()

    image = cv2.imread(image_path)
    cropped_id_card = detect_id_card(image, net, classes, output_layers)

    if cropped_id_card is not None:
        # Process and save the cropped ID card image, e.g., apply perspective transformation or other enhancements
        cv2.imshow("Cropped ID Card", cropped_id_card)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("ID card not detected in the image.")
