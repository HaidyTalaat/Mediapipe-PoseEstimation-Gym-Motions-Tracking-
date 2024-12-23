import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose detector
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose.Pose()  # Import pretrained model pose from mediapipe to perform pose estimation task

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (Shoulder)
    b = np.array(b)  # Mid point  (elbow)
    c = np.array(c)  # End point  (wrist)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Open video capture
cap = cv2.VideoCapture('wide range for uping and down.mp4') 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit processing size
    frame = cv2.resize(frame, dsize=(1280, 720))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Run MediaPipe pose estimation
    results = mp_pose.process(image)
    image.flags.writeable = True

    # Extract landmarks for angle calculation
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Getting coordinates for shoulder, elbow, and wrist for the left and right arms
        left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
        
        right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Calculate angles for both arms
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Check if the angle is within the acceptable range for a correct curl (between 3° and 178°)
        left_curl_status = "Down" if 100 < left_angle < 178 else "UP"
        right_curl_status = "Down" if 100 < right_angle < 178 else "UP"
        
        # The place where the value of angle will appear (mid point where elbow)
        left_elbow_coords = np.multiply(left_elbow, [frame.shape[1], frame.shape[0]]).astype(int)
        right_elbow_coords = np.multiply(right_elbow, [frame.shape[1], frame.shape[0]]).astype(int)
        
        # Ensure coordinates are within bounds (for text visibility)
        left_elbow_coords[0] = min(max(left_elbow_coords[0], 10), frame.shape[1])
        left_elbow_coords[1] = min(max(left_elbow_coords[1], 10), frame.shape[0] )
        right_elbow_coords[0] = min(max(right_elbow_coords[0], 10), frame.shape[1] )
        right_elbow_coords[1] = min(max(right_elbow_coords[1], 10), frame.shape[0] )

        # Display angle and curl status on the frame for both elbows
        cv2.putText(frame, f"Left Angle: {int(left_angle)}", 
                    tuple(left_elbow_coords), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, left_curl_status, 
                    (left_elbow_coords[0], left_elbow_coords[1] +  0), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if left_curl_status == "UP" else (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Right Angle: {int(right_angle)}", 
                    tuple(right_elbow_coords),
                    # (255,255,255) white color for word  Right Angle that will be written 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, right_curl_status, 
                    (right_elbow_coords[0], right_elbow_coords[1] + 30), 
                    # in cv it's BGR so if it's UP (0,255,0)  --> Green
                    #else (0,0,255) -->   Red
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if right_curl_status == "UP" else (0, 0, 255), 2, cv2.LINE_AA)
        print(right_curl_status,' ',left_curl_status)

    except Exception as e:
        print("Error extracting landmarks:", e)

    # Draw landmarks (apply flip for the pose)
    if results.pose_landmarks:
      #  frame = cv2.flip(frame, 1)  # Flip frame for mirror effect on landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # Show the frame with detected pose landmarks
    cv2.imshow('Pose Detection', frame)

    # Exit loop when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()
