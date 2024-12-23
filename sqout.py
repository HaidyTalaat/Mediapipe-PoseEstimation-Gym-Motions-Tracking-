import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose detector
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose.Pose()  # Import pretrained model pose from mediapipe to perform pose estimation task

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (hip)
    b = np.array(b)  # Mid point  (knee)
    c = np.array(c)  # End point  (ankle)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Open video capture
cap = cv2.VideoCapture('sqot_motion.mp4') 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit processing size
    frame = cv2.resize(frame, dsize=(1200,700))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Run MediaPipe pose estimation
    results = mp_pose.process(image)
    image.flags.writeable = True

    # Extract landmarks for angle calculation
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Getting coordinates for hip,knee,ankle for the left and right arms
        
        left_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        right_hip = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Calculate angles for both arms
        left_angle = calculate_angle( left_hip, left_knee,left_ankle)
        right_angle = calculate_angle(right_hip, right_knee,right_ankle)
        
        # Check if the angle is within the acceptable range for a correct curl (between 160° and 180°)
        left_curl_status = "UP" if 160 < left_angle < 180 else "Down"
        right_curl_status = "UP" if 160 < right_angle < 180 else "Down"
        
        # The place where the value of angle will appear (mid point where elbow)
        left_knee_coords = np.multiply(left_knee, [frame.shape[1], frame.shape[0]]).astype(int)
        right_knee_coords = np.multiply(right_knee, [frame.shape[1], frame.shape[0]]).astype(int)
        
        # Ensure coordinates are within bounds (for text visibility)
        left_knee_coords[0] = min(max(left_knee_coords[0], 10), frame.shape[1])
        left_knee_coords[1] = min(max(left_knee_coords[1], 10), frame.shape[0] )
        right_knee_coords[0] = min(max(right_knee_coords[0], 10), frame.shape[1] )
        right_knee_coords[1] = min(max(right_knee_coords[1], 10), frame.shape[0] )

        # Display angle and curl status on the frame for both elbows
        cv2.putText(frame, f"Left {int(left_angle)}", 
                    tuple(left_knee_coords), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, left_curl_status, 
                    (left_knee_coords[0], left_knee_coords[1] +  30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if left_curl_status == "UP" else (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"Right: {int(right_angle)}", 
                    tuple(right_knee_coords),
                    # (255,255,255) white color for word  Right Angle that will be written 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, right_curl_status, 
                    (right_knee_coords[0], right_knee_coords[1]+30), 
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
