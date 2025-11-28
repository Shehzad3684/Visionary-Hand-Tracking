import cv2
import mediapipe as mp
import pymunk
import math
import random
import numpy as np

# --- CONFIGURATION & PALETTE ---
# Cyberpunk Palette (BGR format)
COLOR_NEON_GREEN = (57, 255, 20)
COLOR_HOT_PINK = (180, 105, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_UI_BG = (30, 30, 30) # Dark Gray
COLOR_TEXT = (255, 255, 255)

MAX_BALLS = 25
GRAVITY = 900.0
BALL_RADIUS = 12
PINCH_THRESHOLD = 0.05
MAX_HANDS = 2
WALL_THICKNESS = 8

# --- GAME STATE ---
score = 0
frame_count = 0
targets = [] 
particles = [] # List for explosion particles
balls = []     # List of active balls (with trails)

# --- PHYSICS SETUP ---
space = pymunk.Space()
space.gravity = (0, GRAVITY)

# --- VISION SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=MAX_HANDS, model_complexity=1)

cap = cv2.VideoCapture(0)

# --- CLASSES ---

class GameBall:
    def __init__(self, x, y):
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, BALL_RADIUS)
        self.body = pymunk.Body(mass, moment)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, BALL_RADIUS)
        self.shape.elasticity = 0.85
        self.shape.friction = 0.5
        space.add(self.body, self.shape)
        
        # Trail System
        self.trail = [] # Stores (x,y) tuples
        self.max_trail = 10

    def update_trail(self):
        pos = (int(self.body.position.x), int(self.body.position.y))
        self.trail.append(pos)
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

class Particle:
    """Tiny debris that spawns when a target is hit."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-5, 5)
        self.vy = random.uniform(-5, 5)
        self.life = 1.0 # Starts at 100% opacity
        self.decay = random.uniform(0.02, 0.05)
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay

# --- HELPER FUNCTIONS ---

def create_bone_barrier(p1, p2):
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = (0, 0)
    shape = pymunk.Segment(body, p1, p2, WALL_THICKNESS)
    shape.elasticity = 0.8
    shape.friction = 0.5
    space.add(body, shape)
    return shape, body

def draw_glass_panel(img, x, y, w, h, text_lines=None):
    """Draws a modern transparent UI panel."""
    sub_img = img[y:y+h, x:x+w]
    
    # 1. Dark Overlay
    overlay = np.full_like(sub_img, COLOR_UI_BG, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.4, overlay, 0.6, 0)
    img[y:y+h, x:x+w] = res
    
    # 2. Border
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
    
    # 3. Text
    if text_lines:
        start_y = y + 30
        for line in text_lines:
            cv2.putText(img, line, (x + 15, start_y), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
            start_y += 30

def spawn_explosion(x, y):
    """Spawns 10-15 particles at the given location."""
    for _ in range(15):
        p = Particle(x, y, COLOR_NEON_GREEN)
        particles.append(p)

# --- MAIN LOOP ---

active_barriers = [] 
pinch_states = {} 

while True:
    success, frame = cap.read()
    if not success: continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_count += 1
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- 1. SPAWN TARGETS ---
    if len(targets) < 3 and random.random() < 0.02:
        tx = random.randint(50, w - 50)
        ty = random.randint(100, h // 2)
        targets.append([tx, ty])

    # --- 2. PHYSICS BARRIER CLEANUP ---
    for shape, body in active_barriers:
        space.remove(shape, body)
    active_barriers.clear()

    # --- 3. HAND PROCESSING ---
    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            # --- Draw "Glow" Skeleton ---
            # We draw twice: thick blur first, then thin white
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=COLOR_HOT_PINK, thickness=5, circle_radius=4),
                mp_drawing.DrawingSpec(color=COLOR_HOT_PINK, thickness=5, circle_radius=4)
            )
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=2)
            )
            
            # Create Barriers
            for connection in mp_hands.HAND_CONNECTIONS:
                s = hand_landmarks.landmark[connection[0]]
                e = hand_landmarks.landmark[connection[1]]
                sx, sy = int(s.x * w), int(s.y * h)
                ex, ey = int(e.x * w), int(e.y * h)
                shape, body = create_bone_barrier((sx, sy), (ex, ey))
                active_barriers.append((shape, body))

            # Spawn Logic
            index = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]
            ix, iy = int(index.x * w), int(index.y * h)
            dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
            
            was_pinching = pinch_states.get(hand_id, False)
            if dist < PINCH_THRESHOLD:
                if not was_pinching: 
                    # Spawn GameBall wrapper
                    new_game_ball = GameBall(ix, iy + 40)
                    balls.append(new_game_ball)
                    if len(balls) > MAX_BALLS:
                        # Remove old ball physics and object
                        old = balls.pop(0)
                        space.remove(old.shape, old.body)
                    pinch_states[hand_id] = True
            else:
                pinch_states[hand_id] = False

    # --- 4. PHYSICS UPDATE ---
    for _ in range(10): space.step(1/600.0)

    # --- 5. RENDER GAME OBJECTS ---

    # A. Draw Trails & Balls
    for b in balls:
        b.update_trail()
        # Draw Trail
        if len(b.trail) > 1:
            # Draw lines connecting trail points
            for i in range(len(b.trail) - 1):
                # Fade alpha... OpenCV doesn't do alpha lines easily, so we change thickness
                thickness = int(math.sqrt(i + 1))
                cv2.line(frame, b.trail[i], b.trail[i+1], COLOR_CYAN, thickness)
        
        # Draw Ball Body
        pos = (int(b.body.position.x), int(b.body.position.y))
        if pos[1] > h + 50: continue
        cv2.circle(frame, pos, BALL_RADIUS, COLOR_CYAN, -1)
        cv2.circle(frame, pos, BALL_RADIUS, (255, 255, 255), 2)

    # B. Targets (Pulsing Effect)
    # Use sin wave for "breathing" size
    pulse = math.sin(frame_count * 0.1) * 3 # Oscillates between -3 and +3
    current_radius = int(25 + pulse)
    
    for i in range(len(targets) - 1, -1, -1):
        tx, ty = targets[i]
        
        # Draw Target
        cv2.circle(frame, (tx, ty), current_radius, COLOR_NEON_GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, (tx, ty), 5, COLOR_NEON_GREEN, -1)
        
        # Check Collision
        hit = False
        for b in balls:
            bx, by = b.body.position
            if math.hypot(bx-tx, by-ty) < (BALL_RADIUS + current_radius):
                hit = True
                break
        
        if hit:
            spawn_explosion(tx, ty)
            targets.pop(i)
            score += 100

    # C. Particles (Explosions)
    for p in particles[:]: # Iterate copy to allow removal
        p.update()
        if p.life <= 0:
            particles.remove(p)
            continue
        # Draw particle (fading size)
        cv2.circle(frame, (int(p.x), int(p.y)), int(p.size * p.life), p.color, -1)

    # --- 6. UI OVERLAY ---
    # Top Left: Score
    draw_glass_panel(frame, 20, 20, 200, 80, [
        f"SCORE: {score}",
        f"BALLS: {len(balls)}"
    ])

    # Bottom Right: Instructions
    draw_glass_panel(frame, w - 240, h - 100, 220, 80, [
        "PINCH: Spawn",
        "PALM: Bounce"
    ])

    cv2.imshow('Neon Bubble Popper Pro', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()