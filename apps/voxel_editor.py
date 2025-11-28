import cv2
import mediapipe as mp
import math
import numpy as np

# --- CONFIGURATION ---
PINCH_THRESHOLD = 0.1
BLOCK_SIZE = 50         # World Size of the block
GRID_COLOR = (255, 0, 0) 
SMOOTHING_SPEED = 0.2

# Camera Settings
FOCAL_LENGTH = 600      # Controls the "Zoom" or Field of View
CAMERA_Z_OFFSET = 400   # How far "back" the camera is from the grid plane

# --- STATE VARIABLES ---
voxel_map = set()
smooth_x = 0
smooth_y = 0

# --- SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# --- 3D MATH FUNCTIONS ---
def project_point(x, y, z, cx, cy):
    """
    Converts a 3D world point (x, y, z) into a 2D screen point (u, v).
    Perspective Division: u = x * (focal_length / z)
    """
    # Prevent division by zero if z is too close
    if z + CAMERA_Z_OFFSET <= 0.1:
        return None

    # Calculate the Perspective Scale
    # The further away (higher z), the smaller this number becomes.
    scale = FOCAL_LENGTH / (z + CAMERA_Z_OFFSET)
    
    # Project and shift to center of screen
    u = int(x * scale + cx)
    v = int(y * scale + cy)
    
    return (u, v)

def draw_perspective_cube(img, x, y, z, size, color, thickness, cx, cy):
    """Draws a cube with true perspective projection."""
    
    # Define the 8 corners of the cube relative to its center (x,y,z)
    s = size / 2
    corners = [
        (x-s, y-s, z-s), (x+s, y-s, z-s), (x+s, y+s, z-s), (x-s, y+s, z-s), # Front Face
        (x-s, y-s, z+s), (x+s, y-s, z+s), (x+s, y+s, z+s), (x-s, y+s, z+s)  # Back Face
    ]

    # Project all 8 corners to 2D screen space
    proj_points = []
    for px, py, pz in corners:
        pt = project_point(px, py, pz, cx, cy)
        if pt is None: return # Skip if behind camera
        proj_points.append(pt)

    # Draw Edges connecting the corners
    # List of pairs of indices that make up the 12 edges of a cube
    edges = [
        (0,1), (1,2), (2,3), (3,0), # Front Face
        (4,5), (5,6), (6,7), (7,4), # Back Face
        (0,4), (1,5), (2,6), (3,7)  # Connecting Lines
    ]

    for s, e in edges:
        cv2.line(img, proj_points[s], proj_points[e], color, thickness)

print("Press 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    h, w, _ = frame.shape 
    
    # Center of the screen (Principal Point)
    cx, cy = w // 2, h // 2

    # --- DRAW SAVED BLOCKS (True 3D!) ---
    for voxel in voxel_map:
        vx, vy = voxel
        # Convert Grid Index -> World Position
        # We center the grid at (0,0) by subtracting half the block size
        world_x = (vx * BLOCK_SIZE)
        world_y = (vy * BLOCK_SIZE)
        world_z = 0 # All saved blocks are on the Z=0 plane for now
        
        draw_perspective_cube(frame, world_x, world_y, world_z, BLOCK_SIZE, GRID_COLOR, 2, cx, cy)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            distance = math.hypot(index.x - thumb.x, index.y - thumb.y)

            # Get Hand Position relative to center of screen
            # (0, 0) is now the center of the video, not top-left
            hand_x = (thumb.x + index.x) / 2 * w - cx
            hand_y = (thumb.y + index.y) / 2 * h - cy
            
            # --- LOGIC ---
            # Snap to nearest block size
            target_grid_x = round(hand_x / BLOCK_SIZE) * BLOCK_SIZE
            target_grid_y = round(hand_y / BLOCK_SIZE) * BLOCK_SIZE

            # Smoothing
            smooth_x += (target_grid_x - smooth_x) * SMOOTHING_SPEED
            smooth_y += (target_grid_y - smooth_y) * SMOOTHING_SPEED

            save_grid_x = round(hand_x / BLOCK_SIZE)
            save_grid_y = round(hand_y / BLOCK_SIZE)

            # --- DRAW CURSOR ---
            if distance < PINCH_THRESHOLD:
                draw_perspective_cube(frame, smooth_x, smooth_y, 0, BLOCK_SIZE, (0, 255, 0), 4, cx, cy)
                voxel_map.add((save_grid_x, save_grid_y))
            else:
                draw_perspective_cube(frame, smooth_x, smooth_y, 0, BLOCK_SIZE, (0, 255, 255), 2, cx, cy)

    cv2.imshow('BoxelXR 3D Perspective', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()