import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import math
import os
import pickle

class JutsuDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize Face Mesh for mouth tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)
        
        # Initialize Selfie Segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # State tracking for jutsu holds
        self.current_jutsu = None
        self.active_jutsu = None
        self.jutsu_start_time = 0
        self.jutsu_hold_duration = 0.5  # Time in seconds to hold the seal
        self.last_known_mid_x = 0
        self.last_known_mid_y = 0
        
        # Load VFX images
        # We assume they are in a 'vfx' folder in the same directory as the script
        self.vfx_images = {
            "Shadow Clone": self.load_vfx("vfx/smoke.png"),
            "Tiger": self.load_vfx("vfx/fire.png"),
            "Chidori": self.load_vfx("vfx/lightning.png"),
            "Rasengan": self.load_vfx("vfx/rasengan.png")
        }
        
        # Video captures for animated VFX
        self.vfx_videos = {}
        if os.path.exists("vfx/rasengen.mp4"):
            self.vfx_videos["Rasengan"] = cv2.VideoCapture("vfx/rasengen.mp4")
        if os.path.exists("vfx/tiger.mov"):
            self.vfx_videos["Tiger"] = cv2.VideoCapture("vfx/tiger.mov")
            
        # Load ML model if it exists
        self.gesture_model = None
        if os.path.exists('gesture_model.pkl'):
            with open('gesture_model.pkl', 'rb') as f:
                self.gesture_model = pickle.load(f)
                print("Loaded custom gesture ML model!")

    def load_vfx(self, path):
        """Loads a transparent PNG if it exists."""
        if os.path.exists(path):
            # Load image with alpha channel (IMREAD_UNCHANGED)
            return cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(f"Warning: VFX image not found at {path}")
        return None

    def calculate_distance(self, p1, p2):
        """Calculates 3D Euclidean distance between two mediapipe landmarks."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def is_finger_curled(self, hand_landmarks, finger_tip_idx, finger_pip_idx, wrist):
        """Checks if a finger is curled by comparing tip distance to wrist vs pip distance to wrist."""
        tip = hand_landmarks.landmark[finger_tip_idx]
        pip = hand_landmarks.landmark[finger_pip_idx]
        
        dist_tip_wrist = self.calculate_distance(tip, wrist)
        dist_pip_wrist = self.calculate_distance(pip, wrist)
        
        # If the tip is closer to the wrist than the PIP joint, it's curled towards the wrist
        return dist_tip_wrist < dist_pip_wrist

    def detect_open_palm(self, results):
        """
        Open palm (Release):
        - At least 1 hand present
        - All fingers extended (not curled)
        """
        if not results.multi_hand_landmarks:
            return False
            
        for hand in results.multi_hand_landmarks:
            wrist = hand.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Check if index, middle, ring, and pinky are extended
            if (not self.is_finger_curled(hand, 8, 6, wrist) and
                not self.is_finger_curled(hand, 12, 10, wrist) and
                not self.is_finger_curled(hand, 16, 14, wrist) and
                not self.is_finger_curled(hand, 20, 18, wrist)):
                return True
                
        return False

    def detect_shadow_clone(self, results):
        """
        Shadow Clone seal (Cross seal):
        - 2 hands present
        - Index fingers (8) intersecting/close
        - Middle, Ring, Pinky curled
        """
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
            return False
            
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
        
        # Index finger tips
        idx_tip1 = hand1.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        idx_tip2 = hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Distance between index finger tips (threshold depends on depth, but <0.08 is usually intersecting/touching)
        dist_idx = self.calculate_distance(idx_tip1, idx_tip2)
        
        if dist_idx > 0.08: # Not close enough to be a cross
            return False
            
        # Check if other fingers are curled for both hands
        for hand in [hand1, hand2]:
            wrist = hand.landmark[self.mp_hands.HandLandmark.WRIST]
            # Middle (12, 10), Ring (16, 14), Pinky (20, 18)
            if not (self.is_finger_curled(hand, 12, 10, wrist) and 
                    self.is_finger_curled(hand, 16, 14, wrist) and 
                    self.is_finger_curled(hand, 20, 18, wrist)):
                return False
                
        return True

    def detect_tiger(self, results):
        """
        Tiger seal:
        - 2 hands present
        - Hands clasped (wrists close)
        - Index fingers (8) and Thumbs (4) straight up and touching
        - Other fingers curled
        """
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
            return False
            
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
        
        wrist1 = hand1.landmark[self.mp_hands.HandLandmark.WRIST]
        wrist2 = hand2.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Check if hands are clasped close together
        if self.calculate_distance(wrist1, wrist2) > 0.2:
            return False
            
        # Index finger tips touching
        idx_tip1 = hand1.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        idx_tip2 = hand2.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        if self.calculate_distance(idx_tip1, idx_tip2) > 0.08:
            return False
            
        # Thumb tips touching
        thumb_tip1 = hand1.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_tip2 = hand2.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        if self.calculate_distance(thumb_tip1, thumb_tip2) > 0.1:
            return False
            
        # Check if index fingers are pointing straight up (tip y < pip y) 
        # Note: In OpenCV image coords, 0 is at the top, so a smaller y is "higher"
        for hand in [hand1, hand2]:
            idx_tip = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            idx_pip = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            if idx_tip.y > idx_pip.y: # Pointing down instead of up
                return False
                
            # Check Middle, Ring, Pinky are curled
            wrist = hand.landmark[self.mp_hands.HandLandmark.WRIST]
            if not (self.is_finger_curled(hand, 12, 10, wrist) and 
                    self.is_finger_curled(hand, 16, 14, wrist) and 
                    self.is_finger_curled(hand, 20, 18, wrist)):
                return False

        return True

    def detect_chidori(self, results):
        """
        Chidori:
        - 2 hands present
        - One hand grabbing the other hand's wrist
        """
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
            return False
            
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
        
        # We can check distance between one hand's palm (middle finger mcp) and other's wrist
        h1_mcp = hand1.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        h2_wrist = hand2.landmark[self.mp_hands.HandLandmark.WRIST]
        dist1 = self.calculate_distance(h1_mcp, h2_wrist)
        
        h2_mcp = hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        h1_wrist = hand1.landmark[self.mp_hands.HandLandmark.WRIST]
        dist2 = self.calculate_distance(h2_mcp, h1_wrist)
        
        # If either hand is holding the other's wrist
        if dist1 < 0.08 or dist2 < 0.08:
            return True
            
        return False

    def detect_rasengan(self, results):
        """
        Rasengan:
        - 2 hands present
        - Vertical Alignment: Hand 1 Palm Center (9) and Hand 2 Palm Center (9) share almost exact X
        - Gap: Distinct moderate distance on Y-axis
        """
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
            return False
            
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
        
        # Landmark 9 is MIDDLE_FINGER_MCP
        h1_mcp = hand1.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        h2_mcp = hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Check X alignment (should be tightly aligned vertically)
        if abs(h1_mcp.x - h2_mcp.x) > 0.1:
            return False
            
        # Check Y gap (must have a gap for the sphere to form)
        y_dist = abs(h1_mcp.y - h2_mcp.y)
        if y_dist < 0.05 or y_dist > 0.4:
            return False
            
        return True

    def overlay_image_alpha(self, img, img_overlay, x, y, alpha_mask):
        """Overlays a transparent PNG image onto the frame."""
        # Calculate image ranges to avoid crashing if the overlay goes out of bounds
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        # Blend overlay within the determined ranges using alpha mask
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis] / 255.0

        alpha_inv = 1.0 - alpha

        img_crop[:] = alpha * img_overlay_crop[:, :, :3] + alpha_inv * img_crop

    def process_frame(self, frame):
        """Processes a single frame: detects hands, evaluates jutsus, applies VFX."""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_jutsu = None
        open_palm_detected = False
        
        # Draw landmarks and evaluate logic
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
            
            # Use ML model if available, else fallback to heuristic rules
            if self.gesture_model is not None:
                row = {}
                sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
                
                for i in range(2):
                    if i < len(sorted_hands):
                        hand = sorted_hands[i]
                        for j, lm in enumerate(hand.landmark):
                            row[f'h{i}_l{j}_x'] = lm.x
                            row[f'h{i}_l{j}_y'] = lm.y
                            row[f'h{i}_l{j}_z'] = lm.z
                    else:
                        for j in range(21):
                            row[f'h{i}_l{j}_x'] = 0.0
                            row[f'h{i}_l{j}_y'] = 0.0
                            row[f'h{i}_l{j}_z'] = 0.0
                            
                df_row = pd.Series(row)
                coords = df_row.values.reshape(-1, 3)
                valid_mask = np.any(coords != 0, axis=1)
                
                if np.any(valid_mask):
                    valid_coords = coords[valid_mask]
                    centroid = np.mean(valid_coords, axis=0)
                    translated_coords = np.copy(coords)
                    translated_coords[valid_mask] -= centroid
                    distances = np.linalg.norm(translated_coords[valid_mask], axis=1)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        translated_coords[valid_mask] /= max_dist
                        
                    normalized_features = translated_coords.flatten()
                    df_features = pd.DataFrame([normalized_features], columns=df_row.index)
                    pred = self.gesture_model.predict(df_features)[0]
                    
                    if pred.lower() in ["open palm", "release", "none", "stop"]:
                        open_palm_detected = True
                    else:
                        detected_jutsu = pred
            else:
                # Fallback to hard-coded rules
                if self.detect_open_palm(results):
                    open_palm_detected = True
                elif self.detect_shadow_clone(results):
                    detected_jutsu = "Shadow Clone"
                elif self.detect_tiger(results):
                    detected_jutsu = "Tiger"
                elif self.detect_chidori(results):
                    detected_jutsu = "Chidori"
                elif self.detect_rasengan(results):
                    detected_jutsu = "Rasengan"
                
        if open_palm_detected:
            self.active_jutsu = None
            self.current_jutsu = None
        else:
            # State machine for holding a jutsu over time
            if detected_jutsu:
                if self.current_jutsu == detected_jutsu:
                    # Same jutsu detected, check duration
                    elapsed = time.time() - self.jutsu_start_time
                    if elapsed >= self.jutsu_hold_duration:
                        self.active_jutsu = detected_jutsu
                else:
                    # New jutsu just started
                    self.current_jutsu = detected_jutsu
                    self.jutsu_start_time = time.time()
            else:
                # No valid jutsu, reset current tracking state
                self.current_jutsu = None
                
        # Apply the active jutsu effect if there is one
        if self.active_jutsu:
            self.apply_vfx(frame, results, self.active_jutsu)
            
        # Display detection status in the corner
        display_jutsu = self.active_jutsu if self.active_jutsu else (self.current_jutsu if self.current_jutsu else 'None')
        status_text = f"Seal: {display_jutsu}"
        color = (0, 255, 0) if self.active_jutsu else ((0, 255, 255) if self.current_jutsu else (0, 0, 255))
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
        return frame

    def apply_vfx(self, frame, results, jutsu_name):
        """Calculates dynamic positioning and draws the VFX overlay."""
        h, w, _ = frame.shape
        lower_name = jutsu_name.lower()

        if "shadow clone" in lower_name or "clone" in lower_name:
            # --- SHADOW CLONE EFFECT WITH SEGMENTATION ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            seg_results = self.selfie_segmentation.process(rgb_frame)
            mask = seg_results.segmentation_mask
            
            if mask is None:
                return

            # Smooth mask
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            
            # Replicate the JS clones with translation and scaling
            clones = [
                {'x': -int(w*0.35), 'y': 0, 'scale': 0.85, 'alpha': 0.6},
                {'x': int(w*0.35), 'y': 0, 'scale': 0.85, 'alpha': 0.6},
                {'x': -int(w*0.18), 'y': 0, 'scale': 0.95, 'alpha': 0.8},
                {'x': int(w*0.18), 'y': 0, 'scale': 0.95, 'alpha': 0.8}
            ]
            
            original_frame = frame.copy()
            
            for clone in clones:
                scale = clone['scale']
                tx = clone['x']
                ty = clone['y']
                
                # Keep scaling centered
                adjusted_tx = tx + w * (1 - scale) / 2
                adjusted_ty = ty + h * (1 - scale) / 2
                
                M = np.float32([[scale, 0, adjusted_tx], [0, scale, adjusted_ty]])
                
                shifted_frame = cv2.warpAffine(original_frame, M, (w, h))
                shifted_mask = cv2.warpAffine(mask, M, (w, h))
                
                # Apply alpha
                effective_alpha = np.clip(shifted_mask, 0, 1) * clone['alpha']
                effective_alpha_3d = np.stack((effective_alpha,) * 3, axis=-1)
                
                # Blend clone onto frame
                frame[:] = (frame * (1 - effective_alpha_3d) + shifted_frame * effective_alpha_3d).astype(np.uint8)
                
            # Draw the original person back on top so they are in front
            effective_alpha_orig = np.clip(mask, 0, 1)
            effective_alpha_orig_3d = np.stack((effective_alpha_orig,) * 3, axis=-1)
            frame[:] = (frame * (1 - effective_alpha_orig_3d) + original_frame * effective_alpha_orig_3d).astype(np.uint8)
            
            # Add dramatic text
            cv2.putText(frame, "SHADOW CLONE JUTSU!", (w // 2 - 200, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            return

        if "chidori" in lower_name:
            # --- CHIDORI EFFECT (LIGHTNING) ---
            vfx_img = self.vfx_images.get(jutsu_name) or self.vfx_images.get("Chidori")
            
            if results and results.multi_hand_landmarks:
                # Default to first hand
                hand = results.multi_hand_landmarks[0]
                
                # If there are 2 hands, figure out which one is being grabbed
                if len(results.multi_hand_landmarks) == 2:
                    h1 = results.multi_hand_landmarks[0]
                    h2 = results.multi_hand_landmarks[1]
                    
                    h1_mcp = h1.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    h2_wrist = h2.landmark[self.mp_hands.HandLandmark.WRIST]
                    dist1 = self.calculate_distance(h1_mcp, h2_wrist)
                    
                    h2_mcp = h2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    h1_wrist = h1.landmark[self.mp_hands.HandLandmark.WRIST]
                    dist2 = self.calculate_distance(h2_mcp, h1_wrist)
                    
                    # If h1 is grabbing h2's wrist, then h2 is the active hand holding the Chidori
                    if dist1 < dist2:
                        hand = h2
                    else:
                        hand = h1

                # Place it at the base of the fingers instead of the wrist
                target_lm = hand.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                px, py = int(target_lm.x * w), int(target_lm.y * h)
                
                if vfx_img is None:
                    # Procedural intense lightning effect (Chidori)
                    
                    # 1. Intense Core Glow (hot white)
                    cv2.circle(frame, (px, py), 20, (255, 255, 255), -1)
                    
                    # 2. Chaotic inner dense lightning sphere
                    for _ in range(60): # Lots of short lines for the dense core web
                        sx = px + np.random.randint(-35, 35)
                        sy = py + np.random.randint(-35, 35)
                        ex = sx + np.random.randint(-20, 20)
                        ey = sy + np.random.randint(-20, 20)
                        color = (255, np.random.randint(230, 255), 150) # Ice blue/white
                        cv2.line(frame, (sx, sy), (ex, ey), color, np.random.randint(1, 4))
                            
                    # 3. Long outward lightning tendrils
                    for _ in range(25): # Number of main bolts
                        sx, sy = px, py
                        theta = np.random.uniform(0, 2 * np.pi)
                        thickness = np.random.randint(3, 7) # Thicker base
                        segments = np.random.randint(4, 9) # Longer reach
                        for j in range(segments):
                            r = np.random.randint(20, 55) # segment length
                            theta += np.random.uniform(-0.5, 0.5) # jitter angle
                            ex = int(sx + r * np.cos(theta))
                            ey = int(sy + r * np.sin(theta))
                            
                            # Occasional branching
                            if np.random.random() > 0.6:
                                branch_theta = theta + np.random.uniform(-1, 1)
                                br = np.random.randint(15, 40)
                                bx = int(sx + br * np.cos(branch_theta))
                                by = int(sy + br * np.sin(branch_theta))
                                branch_thick = max(1, thickness - 2)
                                branch_color = (255, np.random.randint(200, 255), 100)
                                cv2.line(frame, (sx, sy), (bx, by), branch_color, branch_thick)
                                
                            color = (255, np.random.randint(220, 255), 50) # Cyan/blue
                            cv2.line(frame, (sx, sy), (ex, ey), color, thickness)
                            sx, sy = ex, ey
                            thickness = max(1, thickness - 1)
                        
                    cv2.putText(frame, "CHIDORI!", (px - 80, py + 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 240, 100), 4)
                else:
                    vfx_h, vfx_w = vfx_img.shape[:2]
                    start_x = px - (vfx_w // 2)
                    start_y = py - (vfx_h // 2)
                    alpha_channel = vfx_img[:, :, 3]
                    self.overlay_image_alpha(frame, vfx_img, start_x, start_y, alpha_channel)
            return

        if "rasengan" in lower_name or "rasen" in lower_name:
            # --- RASENGAN EFFECT (SPINNING SPHERE) ---
            vfx_img = self.vfx_images.get(jutsu_name) or self.vfx_images.get("Rasengan")
            
            if results and results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
                hand1 = results.multi_hand_landmarks[0]
                hand2 = results.multi_hand_landmarks[1]
                
                h1_mcp = hand1.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                h2_mcp = hand2.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                
                # Calculate exact midpoint between the two palms
                mid_x = int(((h1_mcp.x + h2_mcp.x) / 2) * w)
                mid_y = int(((h1_mcp.y + h2_mcp.y) / 2) * h)
                
                # Dynamically calculate the size of the Rasengan based on the gap between hands
                hand_dist = int(abs(h1_mcp.y - h2_mcp.y) * h)
                # Make it even bigger and distinctly wider (oval/disk shape)
                target_w = max(800, int(hand_dist * 4.0))
                target_h = max(600, int(hand_dist * 3.0))
                
                vfx_cap = getattr(self, 'vfx_videos', {}).get("Rasengan")
                if vfx_cap and vfx_cap.isOpened():
                    success, vfx_frame = vfx_cap.read()
                    if not success:
                        # Loop video
                        vfx_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        success, vfx_frame = vfx_cap.read()
                        
                    if success:
                        # Resize video frame dynamically to be massive and wide
                        vfx_frame = cv2.resize(vfx_frame, (target_w, target_h))
                        vfx_h, vfx_w = vfx_frame.shape[:2]
                        
                        start_x = mid_x - (vfx_w // 2)
                        start_y = mid_y - (vfx_h // 2)
                        
                        # Additive blending to remove black background
                        y1, y2 = max(0, start_y), min(h, start_y + vfx_h)
                        x1, x2 = max(0, start_x), min(w, start_x + vfx_w)
                        
                        y1o, y2o = max(0, -start_y), min(vfx_h, h - start_y)
                        x1o, x2o = max(0, -start_x), min(vfx_w, w - start_x)
                        
                        if y1 < y2 and x1 < x2:
                            roi = frame[y1:y2, x1:x2]
                            overlay = vfx_frame[y1o:y2o, x1o:x2o]
                            # cv2.add saturates at 255, creating a perfect screen blending mode
                            frame[y1:y2, x1:x2] = cv2.add(roi, overlay)
                            
                        cv2.putText(frame, "RASENGAN!", (mid_x - 100, mid_y + int(target_h * 0.45)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 4)
                        return

                if vfx_img is None:
                    # Procedural spinning wind-sphere
                    
                    # Scale based on target dimensions
                    core_r1_w = int(target_w * 0.15)
                    core_r1_h = int(target_h * 0.15)
                    core_r2_w = int(target_w * 0.08)
                    core_r2_h = int(target_h * 0.08)
                    
                    # Core glow (drawn as ellipses to support wide shape)
                    cv2.ellipse(frame, (mid_x, mid_y), (core_r1_w, core_r1_h), 0, 0, 360, (255, 255, 150), -1) # Light cyan
                    cv2.ellipse(frame, (mid_x, mid_y), (core_r2_w, core_r2_h), 0, 0, 360, (255, 255, 255), -1) # White core
                    
                    # Spinning wind arcs
                    t = time.time() * 8 # Speed of spin
                    for i in range(4):
                        angle = int((t * 20 + i * 45) % 360)
                        
                        axes1 = (int(target_w * 0.20), int(target_h * 0.08))
                        axes2 = (int(target_w * 0.08), int(target_h * 0.20))
                        # Draw ellipses to simulate spinning wind streams
                        cv2.ellipse(frame, (mid_x, mid_y), axes1, angle, 0, 360, (255, 200, 50), 3)
                        cv2.ellipse(frame, (mid_x, mid_y), axes2, angle, 0, 360, (255, 200, 50), 3)
                        
                        # Outer thinner energy rings
                        angle_outer = int((-t * 15 + i * 45) % 360)
                        axes_outer = (int(target_w * 0.28), int(target_h * 0.06))
                        cv2.ellipse(frame, (mid_x, mid_y), axes_outer, angle_outer, 0, 360, (255, 255, 200), 2)
                        
                    cv2.putText(frame, "RASENGAN!", (mid_x - 100, mid_y + int(target_h * 0.45)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 4)
                else:
                    vfx_h, vfx_w = vfx_img.shape[:2]
                    start_x = mid_x - (vfx_w // 2)
                    start_y = mid_y - (vfx_h // 2)
                    alpha_channel = vfx_img[:, :, 3]
                    self.overlay_image_alpha(frame, vfx_img, start_x, start_y, alpha_channel)
            return

        if "tiger" in lower_name or "fireball" in lower_name:
            # --- FIREBALL JUTSU (MOUTH TRACKING) ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb_frame)
            
            vfx_img = self.vfx_images.get(jutsu_name) or self.vfx_images.get("Tiger")
            
            # Default to center if no face detected
            mouth_x, mouth_y = w // 2, h // 2
            
            if face_results.multi_face_landmarks:
                face = face_results.multi_face_landmarks[0]
                # Landmark 13 is inner upper lip, 14 is inner lower lip
                mouth_x = int(((face.landmark[13].x + face.landmark[14].x) / 2) * w)
                mouth_y = int(((face.landmark[13].y + face.landmark[14].y) / 2) * h)
                
            vfx_cap = getattr(self, 'vfx_videos', {}).get("Tiger")
            if vfx_cap and vfx_cap.isOpened():
                success, vfx_frame = vfx_cap.read()
                if not success:
                    # Loop video
                    vfx_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, vfx_frame = vfx_cap.read()
                    
                if success:
                    # Make the fireball bigger
                    target_w = 1200
                    target_h = 800
                    vfx_frame = cv2.resize(vfx_frame, (target_w, target_h))
                    vfx_h, vfx_w = vfx_frame.shape[:2]
                    
                    # Align center of canvas near mouth, shifted a little bit to the right
                    offset_x = 300
                    start_x = mouth_x - (vfx_w // 2) + offset_x
                    start_y = mouth_y - (vfx_h // 2)
                    
                    # Additive blending
                    y1, y2 = max(0, start_y), min(h, start_y + vfx_h)
                    x1, x2 = max(0, start_x), min(w, start_x + vfx_w)
                    
                    y1o, y2o = max(0, -start_y), min(vfx_h, h - start_y)
                    x1o, x2o = max(0, -start_x), min(vfx_w, w - start_x)
                    
                    if y1 < y2 and x1 < x2:
                        roi = frame[y1:y2, x1:x2]
                        overlay = vfx_frame[y1o:y2o, x1o:x2o]
                        frame[y1:y2, x1:x2] = cv2.add(roi, overlay)
                        
                    cv2.putText(frame, "FIREBALL JUTSU!", (mouth_x - 150, mouth_y + 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 4)
                    return

            if vfx_img is None:
                # Procedural horizontal fire effect
                # Create a massive horizontal cone of fire shooting to the side
                for _ in range(60): # More particles for a better look
                    offset_x = np.random.randint(20, 400)
                    
                    # Shoot to the right side of the screen
                    px = mouth_x + offset_x
                    
                    # The vertical spread grows larger as it gets further from the mouth (cone shape)
                    spread = max(10, int(offset_x * 0.35))
                    py = mouth_y + np.random.randint(-spread, spread)
                    
                    # Particles get larger as they expand
                    size = np.random.randint(10, 20) + int(offset_x * 0.1)
                    
                    # OpenCV uses BGR, so (0, 100-200, 255) is orange/yellow/red
                    color = (0, np.random.randint(80, 220), 255)
                    cv2.circle(frame, (px, py), size, color, -1)
                    
                cv2.putText(frame, "FIREBALL JUTSU!", (mouth_x - 150, mouth_y + 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 4)
            else:
                vfx_h, vfx_w = vfx_img.shape[:2]
                start_x = mouth_x - (vfx_w // 2)
                # Offset fire slightly upwards so it comes out of mouth
                start_y = mouth_y - (vfx_h // 2) 
                alpha_channel = vfx_img[:, :, 3]
                self.overlay_image_alpha(frame, vfx_img, start_x, start_y, alpha_channel)
            return

        # --- DEFAULT VFX HANDLING ---
        vfx_img = self.vfx_images.get(jutsu_name)
        
        # Update last known midpoint if hands are currently detected
        if results and results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            hand1_wrist = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.WRIST]
            hand2_wrist = results.multi_hand_landmarks[1].landmark[self.mp_hands.HandLandmark.WRIST]
            
            self.last_known_mid_x = int(((hand1_wrist.x + hand2_wrist.x) / 2) * w)
            self.last_known_mid_y = int(((hand1_wrist.y + hand2_wrist.y) / 2) * h)
            
        mid_x = self.last_known_mid_x if self.last_known_mid_x else w // 2
        mid_y = self.last_known_mid_y if self.last_known_mid_y else h // 2

        if vfx_img is None:
            # Fallback text if no PNG is provided
            cv2.putText(frame, f"{jutsu_name.upper()} JUTSU!", (mid_x - 150, mid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
            return
        
        # Position image so it's centered exactly on the wrist midpoint
        vfx_h, vfx_w = vfx_img.shape[:2]
        start_x = mid_x - (vfx_w // 2)
        start_y = mid_y - (vfx_h // 2)
        
        # Extract alpha channel
        alpha_channel = vfx_img[:, :, 3]
        
        # Overlay!
        self.overlay_image_alpha(frame, vfx_img, start_x, start_y, alpha_channel)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Attempt to set highest framerate and resolution possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    detector = JutsuDetector()
    
    print("================================")
    print(" Naruto Jutsu Detector Started! ")
    print("================================")
    print("Press 'q' inside the video window to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam.")
            continue
            
        # Flip frame horizontally for an intuitive "mirror" view
        frame = cv2.flip(frame, 1)
        
        # Process and overlay hand tracking/jutsus
        processed_frame = detector.process_frame(frame)
        
        # Show window
        cv2.imshow('Live Naruto Jutsu Detector', processed_frame)
        
        # Break loop cleanly on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
