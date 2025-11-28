# Visionary Hand Tracking Suite

A collection of computer vision and augmented reality applications powered by **MediaPipe**, **OpenCV**, and **Pymunk**. This project explores advanced hand-tracking concepts including spatial quantization (voxel snapping), true 3D perspective projection, and physics-based interaction.

## Applications Included

### 1. BoxelXR (3D Voxel Editor)
A spatial computing prototype that allows users to build 3D structures using hand gestures.
* **Features:**
    * **True 3D Perspective:** Implements custom focal length and depth projection math to simulate depth on 2D screens.
    * **Spatial Hashing:** Snaps cursor movement to a 3D grid for precise editing.
    * **Smoothing Algorithms:** Uses Linear Interpolation (Lerp) to filter hand tremors for fluid interaction.
    * **Gesture Recognition:** Custom pinch-to-place detection mechanics.

### 2. Neon Bubble Popper (AR Physics Game)
An augmented reality game where the user's hands are rendered as solid physical objects in real-time.
* **Features:**
    * **Real-Time Physics:** Uses `pymunk` to simulate gravity, collision, and friction.
    * **Dynamic Kinematic Bodies:** Converts the hand skeleton (bones) into solid physical walls every frame.
    * **Sub-stepping:** Physics calculations run at 600Hz to prevent object tunneling during fast movements.
    * **Particle Systems:** Custom engine for rendering explosion effects and object motion trails.
    * **Glassmorphism UI:** Modern user interface implementation using semi-transparent overlays.

## Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Visionary-Hand-Tracking.git](https://github.com/YOUR_USERNAME/Visionary-Hand-Tracking.git)
    cd Visionary-Hand-Tracking
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main launcher to choose an application:

```bash
python main.py