Project: AI-Powered Air Canvas – Gesture-Controlled Virtual Drawing using Hand Tracking
(Computer Vision • MediaPipe • OpenCV • Python)
Developed a real-time air drawing application that lets users draw and erase in the air using only their webcam and hand gestures — no physical pen, mouse, or touch required.
Key Features:
Index finger extension → draws smooth lines (orange brush)
Open palm gesture → activates eraser mode (thick black eraser)
Fist / curled fingers → stops drawing
Real-time gesture detection & smooth stroke rendering

Technologies Used:
MediaPipe Hands (Google's deep learning-based hand landmark detection model)
OpenCV for video capture, image processing, drawing & UI overlay
NumPy for coordinate calculations
Python (with deque for smoothing drawing path)

How it works (briefly):
The application captures webcam feed, processes each frame using MediaPipe's HandLandmarker model to detect 21 hand landmarks, analyzes finger tip positions relative to knuckles to recognize gestures, and renders persistent drawing on a virtual canvas blended with the live video.
Learning outcomes / skills demonstrated:
Real-time computer vision & gesture recognition
Working with pre-trained deep learning models in production-like setting
Smooth trajectory interpolation & anti-shaky drawing
Human-Computer Interaction (HCI) principles
OpenCV drawing & blending techniques
