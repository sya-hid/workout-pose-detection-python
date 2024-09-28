import cv2
import mediapipe as mp
import numpy as np
import time

# Inisialisasi MediaPipe Pose dan Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands()

# Kalibrasi
calibration_time = 3  # Tunggu selama 3 detik sebelum mulai mendeteksi
calibration_start_time = None  # Waktu kalibrasi awal
is_calibrating = True  # Status kalibrasi
is_tracking_started = False  # Status tracking

# Fungsi untuk memeriksa apakah seluruh tubuh berada dalam frame
def is_body_in_frame(landmarks):
    buffer = 0.052
    for landmark in landmark_list:
        x = landmarks[landmark.value].x
        y = landmarks[landmark.value].y
        if x < buffer or x > (1 - buffer) or y < buffer or y > (1 - buffer):
            return False
    return True

# Fungsi untuk memastikan kalibrasi selesai sebelum tracking
def start_tracking(landmarks):
    global calibration_start_time, is_calibrating, is_tracking_started
    if is_body_in_frame(landmarks):
        if calibration_start_time is None:
            calibration_start_time = time.time()
        
        elapsed_time = time.time() - calibration_start_time
        if elapsed_time >= calibration_time:
            is_calibrating = False
            is_tracking_started = True
            print("Tracking started.")
    else:
        calibration_start_time = None  # Reset kalibrasi jika tubuh keluar dari frame

# Saat mendeteksi pose
def process_pose(landmarks):
    global is_calibrating, is_tracking_started
    
    if is_calibrating:
        start_tracking(landmarks)
    elif is_tracking_started:
        print("Tracking user pose...")

# Variabel untuk menghitung push-up, plank, squat, jump, dan durasi plank
push_up_count = 0
plank_duration = 0  # Durasi total plank dalam detik
squat_count = 0
situp_count = 0
jump_count = 0
stage_push_up = None
stage_sit_up = None
stage_jump = None
# Variabel untuk menghitung waktu plank
plank_start_time = None
is_planking = False

# Daftar landmark yang dipakai
landmark_list = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]

heel_buffer = []
wrist_buffer = []
elbow_buffer = []
hip_buffer = []
shoulder_buffer = []
knee_buffer = []
foot_index_buffer = []
texts_to_display = []

BUFFER_SIZE = 5

def calculate_angle(a, b, c):
    a = np.array(a)  # Titik pertama
    b = np.array(b)  # Titik kedua (vertex)
    c = np.array(c)  # Titik ketiga

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

def detect_direction(landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    frame_middle_x = 0.5  # x normalisasi MediaPipe

    left_landmarks_in_left_side = (left_shoulder[0] < frame_middle_x) and (left_hip[0] < frame_middle_x)
    right_landmarks_in_right_side = (right_shoulder[0] > frame_middle_x) and (right_hip[0] > frame_middle_x)

    if left_landmarks_in_left_side and right_landmarks_in_right_side:
        return "Depan"
    elif right_shoulder[0] < left_shoulder[0]:
        return "Arah Kanan"
    else:
        return "Arah Kiri"
    
HEEL_THRESHOLD = 0.765

def on_ground(heel_positions, threshold=HEEL_THRESHOLD):
    avg_heel_y = np.mean(heel_positions)
    return avg_heel_y > threshold
video_path = 'videos/PushUp2.mp4'
video_path = 'videos/Push-Up,StandUp,SquatPrisonerJumpCountdown.mp4'
video_path = 'videos/Plank.mp4'
video_path = 'videos/sit up1.mp4'
video_path = 'videos/pushup-1.mp4'
# video dari asset
cap = cv2.VideoCapture(video_path)
# dari kamera
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak bisa membuka video.")
else:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Gagal membaca frame dari video.")
            break

        image = cv2.resize(image, (640, 480))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # Jalankan kalibrasi dan tracking
            process_pose(landmarks)
            
            if is_tracking_started:
                direction = detect_direction(landmarks)
                if(direction=='Arah Kanan'):
                    shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y]
                    foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                    hand_index = [landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].y]
                    
                else:
                    heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                    hand_index = [landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].y]
                
                index_finger_mcp = [landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP.value].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP.value].y]
                
             

                
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                body_angle = calculate_angle(shoulder, hip, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)
                ankle_angle = calculate_angle( knee, ankle,foot_index)
                wrist_angle=calculate_angle(elbow,wrist,hand_index)
                
                heel_buffer.append(heel[1])
                wrist_buffer.append(wrist[1])
                hip_buffer.append(hip[1])
                shoulder_buffer.append(shoulder[1])
                foot_index_buffer.append(foot_index[1])
                knee_buffer.append(knee[1])

                # Pastikan buffer tidak terlalu besar
                if len(heel_buffer) > BUFFER_SIZE:
                    heel_buffer.pop(0)
                if len(wrist_buffer) > BUFFER_SIZE:
                    wrist_buffer.pop(0)
                if len(hip_buffer) > BUFFER_SIZE:
                    hip_buffer.pop(0)
                if len(shoulder_buffer) > BUFFER_SIZE:
                    shoulder_buffer.pop(0)
                if len(foot_index_buffer) > BUFFER_SIZE:
                    foot_index_buffer.pop(0)
                if len(knee_buffer) > BUFFER_SIZE:
                    knee_buffer.pop(0)

                heel_on_ground = on_ground(heel_buffer)
                wrist_on_ground=on_ground(wrist_buffer)
                hip_on_ground=on_ground(hip_buffer)
                shoulder_on_ground=on_ground(shoulder_buffer)
                foot_index_on_ground=on_ground(foot_index_buffer)
                knee_on_ground=on_ground(knee_buffer)
                
                    # Push Up
                if elbow_angle < 90 and hip_angle>150 and wrist_on_ground and foot_index_on_ground:
                    stage_push_up = "down"
                if elbow_angle > 140 and hip_angle>150 and wrist_on_ground and foot_index_on_ground and (stage_push_up=="down"):
                    stage_push_up = "up"
                    push_up_count += 1
                    # print(f"Push-up count: {push_up_count}")
                    
                # Sit Up
                if hip_angle > 95 and shoulder[1]<hip[1] and hip_on_ground and heel_on_ground and foot_index_on_ground:
                    stage_sit_up = "down"
                if hip_angle < 70 and knee[1]<shoulder[1] and stage_sit_up == 'down' and hip_on_ground  and foot_index_on_ground:
                    stage_sit_up = "up"
                    situp_count += 1
                    print(f"situp{ situp_count}")
                
                # Plank
                if elbow[1] > shoulder[1] and abs(elbow[1] - wrist[1]) < 0.05 and 85 < elbow_angle < 140 and 150 < body_angle < 180 and knee[1] < wrist[1] and wrist_on_ground and foot_index_on_ground and knee_on_ground==False:
                    if not is_planking:
                        plank_start_time = time.time()
                        is_planking = True
                        print("Plank started")
                else:
                    if is_planking:
                        plank_end_time = time.time()
                        plank_duration += plank_end_time - plank_start_time
                        is_planking = False
                        print(f"Plank ended. Total Duration: {plank_duration:.2f} seconds")
        
                    # Jika sedang planking, tambahkan durasi
                if is_planking:
                    current_time = time.time()
                    current_plank_duration = current_time - plank_start_time
                    print(f"Planking... Current Duration: {current_plank_duration + plank_duration:.2f} seconds")
                
                # Jump    
                if heel_on_ground and foot_index_on_ground :
                    stage_jump = "down"          
                if ankle[1] < 0.70 and stage_jump == 'down' and heel_on_ground==False and wrist_on_ground==False and foot_index_on_ground==False:
                    stage_jump = "up"
                    jump_count += 1
                    # print(f"Jump count: {jump_count}")
       
                ##
                cv2.putText(image,f"{wrist_angle:.2f}", tuple(np.multiply(wrist, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA) 
                cv2.putText(image,f"{ankle_angle:.2f}", tuple(np.multiply(ankle, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA) 
                cv2.putText(image,f'{body_angle:.2f}', tuple(np.multiply(shoulder, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(image,f'{hip_angle:.2f}', tuple(np.multiply(hip, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, f'{elbow_angle:.2f}', tuple(np.multiply(elbow, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, f'{knee_angle:.2f}', tuple(np.multiply(knee, [640, 480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
               
            activity_order = {}

            if push_up_count != 0 and 'push_up' not in activity_order:
                activity_order['push_up'] = 'Push-up Count'
            if squat_count != 0 and 'squat' not in activity_order:
                activity_order['squat'] = 'Squat Count'
            if plank_duration != 0 and 'plank' not in activity_order:
                activity_order['plank'] = 'Plank Duration'
            if situp_count != 0 and 'situp' not in activity_order:
                activity_order['situp'] = 'Sit Up Count'
            if jump_count != 0 and 'jump' not in activity_order:
                activity_order['jump'] = 'Jump Count'

            if activity_order:
                # Posisi awal Y untuk menampilkan teks
                start_y = 30
                line_height = 30  # Jarak antar teks

                # Loop untuk menampilkan teks sesuai urutan
                for i, (key,label) in enumerate(activity_order.items()):
                    y_position = start_y + i * line_height
                    if key == 'push_up':
                        value = f"{label}: {push_up_count}"
                    elif key == 'squat':
                        value = f"{label}: {squat_count}"
                    elif key == 'plank':
                        value = f"{label}: {plank_duration:.2f} sec"
                    elif key == 'situp':
                        value = f"{label}: {situp_count}"
                    elif key == 'jump':
                        value = f"{label}: {jump_count}"
                    cv2.putText(image, value, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # cv2.putText(image, f"Video diambil dari: {direction}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Pose Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
