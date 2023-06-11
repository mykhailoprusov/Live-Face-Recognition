import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

counter = 0

reference_img = cv2.imread("reference.jpg")

face_match = False
gender = 'Unknown'
age = 'Unknown'

def check_face(frame):
    global face_match, gender , age

    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:

            results = DeepFace.analyze(frame,actions=("gender","age"))

            lock = threading.Lock()

            lock.acquire()
            woman = results[0]['gender']['Woman']
            man = results[0]['gender']['Man']

            if man > woman:
                gender = f'{man:.0f}% Male'
                age = results[0]['age']
            else:
                gender = f'{woman:.0f}% Female'
                age = results[0]['age']

            lock.release()
            face_match = True

        else:
            face_match = False

    except ValueError:
        face_match = False

while True:
    ret,frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()

            except ValueError:
                pass

        counter += 1

        if face_match:
            cv2.putText(frame, f"MATCH! Sex:{gender},Age:{age}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        else:
            cv2.putText(frame, f"NO MATCH! Sex:{gender},Age:{age}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('video',frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()