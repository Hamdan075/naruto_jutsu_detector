import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

def main():
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    data = []
    
    print("================================")
    print(" Jutsu Data Collector ")
    print("================================")
    import sys
    if len(sys.argv) > 1:
        gesture_name = " ".join(sys.argv[1:])
        print(f"Using gesture name from command line: {gesture_name}")
    else:
        gesture_name = input("Enter the name of the gesture you want to record (e.g., 'Shadow Clone', 'Tiger', 'Open Palm'): ").strip()
    
    print(f"\nAwesome. Switch back to the webcam window.")
    print("Press 'r' to start a 10-second countdown before recording 300 frames. Press 'q' to quit.\n")
    
    recording = False
    counting_down = False
    countdown_start_time = 0
    frames_recorded = 0
    max_frames = 300
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            print("Failed to read from webcam.")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw for visual feedback
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # Handle recording logic
        if counting_down:
            elapsed = time.time() - countdown_start_time
            remaining = 10 - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, f"Starting in {remaining}s...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            else:
                counting_down = False
                recording = True
                print("Recording started... hold the pose!")
        elif recording:
            if frames_recorded < max_frames:
                if results.multi_hand_landmarks:
                    # We need to extract up to 2 hands
                    row = {'label': gesture_name}
                    
                    # Sort hands by x-coordinate of wrist so Left Hand is always Hand 1, Right is Hand 2
                    # This ensures the model learns consistently
                    sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
                    
                    for i in range(2):
                        if i < len(sorted_hands):
                            hand = sorted_hands[i]
                            for j, lm in enumerate(hand.landmark):
                                row[f'h{i}_l{j}_x'] = lm.x
                                row[f'h{i}_l{j}_y'] = lm.y
                                row[f'h{i}_l{j}_z'] = lm.z
                        else:
                            # Pad with zeros if second hand is missing (one-handed jutsus)
                            for j in range(21):
                                row[f'h{i}_l{j}_x'] = 0.0
                                row[f'h{i}_l{j}_y'] = 0.0
                                row[f'h{i}_l{j}_z'] = 0.0
                                
                    data.append(row)
                    frames_recorded += 1
                    
                    # Draw progress
                    cv2.putText(frame, f"Recording: {frames_recorded}/{max_frames}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                print("\nRecording finished! Saving to CSV...")
                df = pd.DataFrame(data)
                csv_file = 'gestures.csv'
                
                # Append if exists, write if not
                if os.path.exists(csv_file):
                    df.to_csv(csv_file, mode='a', header=False, index=False)
                else:
                    df.to_csv(csv_file, mode='w', header=True, index=False)
                    
                print(f"Saved {max_frames} frames of '{gesture_name}' to {csv_file}")
                print("Run the script again to record another gesture, or run train_model.py!")
                break
        else:
            cv2.putText(frame, "Press 'r' for 10s countdown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Target: {gesture_name}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
        cv2.imshow('Data Collector', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and not recording and not counting_down:
            counting_down = True
            countdown_start_time = time.time()
            print("Countdown started... get into position!")
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
