import cv2
import mediapipe as mp

# distance euclidian
def distance_euclid(point1, point2):
    return ((point1.x-point2.x)**2+(point1.y-point2.y)**2)**0.5

# mp_solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands = 2) as hands:    
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # dimensoes da imagem
    h, w, _ = image.shape

    # lista para guardar a posiçao das maos
    point_coord = []

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for id, cord in enumerate(hand_landmarks.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                point_coord.append((cx,cy))
        
        count = 0

        # euclidian distance for each finger 
        for nb_hand in range(len(results.multi_hand_landmarks)):
            hand = results.multi_hand_landmarks[nb_hand]
            fingers_open = [distance_euclid(hand.landmark[4], hand.landmark[17]) > distance_euclid(hand.landmark[3], hand.landmark[17]), #Thumb
            distance_euclid(hand.landmark[8], hand.landmark[0]) > distance_euclid(hand.landmark[7], hand.landmark[0]), #Index
            distance_euclid(hand.landmark[12], hand.landmark[0]) > distance_euclid(hand.landmark[11], hand.landmark[0]), #Middle finger
            distance_euclid(hand.landmark[16], hand.landmark[0]) > distance_euclid(hand.landmark[15], hand.landmark[0]), #Ring finger
            distance_euclid(hand.landmark[20], hand.landmark[0]) > distance_euclid(hand.landmark[19], hand.landmark[0]), #Little finger
        ]  
            count += sum(fingers_open)

        # purple retangle 
        cv2.rectangle(image, (80, 10), (300,110), (214, 0, 134), -1) #posiçao, fundo, completo
        cv2.putText(image,str(count),(100,100),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),5)
        #print(cont)

        

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
