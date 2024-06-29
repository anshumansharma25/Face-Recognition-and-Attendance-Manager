import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

aryan_image = face_recognition.load_image_file("images/aryan.jpg")
aryan_encoding = face_recognition.face_encodings(aryan_image)[0]

anshuman_image = face_recognition.load_image_file("images/anshuman.jpg")
anshuman_encoding = face_recognition.face_encodings(anshuman_image)[0]

shivay_image = face_recognition.load_image_file("images/shivay.jpg")
shivay_encoding = face_recognition.face_encodings(shivay_image)[0]

sourav_image = face_recognition.load_image_file("images/sourav.jpg")
sourav_encoding = face_recognition.face_encodings(sourav_image)[0]

arin_image = face_recognition.load_image_file("images/arin.jpg")
arin_encoding = face_recognition.face_encodings(arin_image)[0]

shweta_image = face_recognition.load_image_file("images/shweta.jpg")
shweta_encoding = face_recognition.face_encodings(shweta_image)[0]

known_face_encoding = [
     aryan_encoding,
    anshuman_encoding,
      shivay_encoding,
      sourav_encoding,
    arin_encoding,
    shweta_encoding
]

known_faces_names = [
     "Aryan Yadav",
    "Anshuman Sharma",
     "Shivay Yadav",
      "Sourav Patidar",
    "Arin Laad",
    "Shweta Mam"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1
                fontColor = (0, 255, 0)
                thickness = 3
                lineType = 2

                cv2.putText(frame, name + ' Present',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
