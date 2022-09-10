import cv2 as cv

while(True):
  cap = cv.VideoCapture(0)
  ret, frame = cap.read()

  if ret:
    print(frame)
    cv.imshow("Frame", frame)
  
  if cv.waitKey(1) == ord('q'):
    cap.release()
    cv.destroyAllWindows()
