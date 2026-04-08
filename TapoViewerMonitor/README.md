# TapoViewer

A Java Swing application to view Tapo cameras via RTSP and control them via ONVIF.

## Key Features
- **RTSP Video Streaming:** Powered by **JavaCV (FFmpeg)**. This provides high-performance, low-latency streaming and is guaranteed to work if `ffplay` works on your system.
- **Person Detection & Tracking:** Real-time human shape identification using **OpenCV HOG**. Once detected, individuals are tracked frame-to-frame using **Intersection over Union (IoU)** logic to maintain a consistent history of their movement.
- **Epileptic Seizure Detection:** Analyzes the "internal motion" of tracked individuals to identify rhythmic jerking patterns typical of seizures, highlighting them with a red box and warning label.
- **Static Object Filtering:** Prevents false positives by automatically identifying and hiding stationary objects (like chair legs) that may initially resemble a human shape.
- **Recent Detections Review:** Automatically captures and stores the last 20 person detections (cropped to the person) for quick review via the "Recent Detections" panel.
- **ONVIF PTZ Control:** Uses the modern, asynchronous `onvif-java` library to provide Pan, Tilt, and Zoom functionality.
- **Dynamic Camera Selection:** Organise cameras by "House" and "Name" via an XML configuration file.
- **Modern Java:** Built for Java 17+ (verified with OpenJDK 23.0.1 environment).

## How Person Detection Works
The person detection in **TapoViewerMonitor** uses the **Histogram of Oriented Gradients (HOG)** algorithm, a classic and highly effective method for identifying human shapes in images.

### The Detector (`hog`)
The `HOGDescriptor` is the "brain" of the detection. It doesn't look at individual pixels; instead, it looks at the **gradients** (the direction and intensity of color changes) in small blocks of the image. Humans have a very distinct "gradient signature"—a head-and-shoulders outline, vertical legs, etc. The detector uses a pre-trained **Support Vector Machine (SVM)** model that has already "seen" thousands of images of people and knows exactly what those gradient patterns look like.

### The Search (`detectMultiScale`)
This is the most critical part of the process. It performs a "sliding window" search across the entire image:
- **Multi-Scale:** People can be close to the camera (large) or far away (small). `detectMultiScale` automatically resizes the image multiple times (creating an "image pyramid") and searches each version. This ensures it finds a person whether they are 2 feet or 20 feet from the camera.
- **Detections:** It returns a list of coordinates `(x, y, width, height)` for every person it found.

### Performance & Accuracy Optimizations
- **Downscaling:** Instead of searching the raw high-resolution (2K/4K) Tapo stream directly, the frame is resized to **640 pixels wide**. This allows the CPU to process the frame in milliseconds, keeping the video smooth and the UI responsive.
- **Coordinate Mapping:** Since detection is performed on a downscaled image but displayed on the full UI, the application calculates a scaling ratio to ensure the green tracking box is drawn precisely around the person in the actual display area.

## How Seizure Detection Works
The seizure detection system uses a combination of temporal tracking, motion quantification, and frequency analysis to distinguish between normal activity and rhythmic seizure movements.

### The `TrackedPerson` Class
This class is the core of the analysis. It maintains the state and history for every individual currently seen by the camera:
- **Motion History:** A circular buffer (`LinkedList<Double>`) that stores the "Motion Intensity" of the person over the last **60 frames** (approx. 2-3 seconds).
- **Previous Frame State:** Stores a grayscale version of the person's bounding box from the previous frame to calculate frame-to-frame motion.
- **Cumulative Motion:** Tracks total movement over the person's entire lifetime to help differentiate between living beings and static furniture.

### 1. Motion Quantification (Optical Flow)
The system doesn't just look at if the *box* moves, but if the *pixels inside* the box move.
- **Farneback Optical Flow:** For every frame, the system compares the current pixels of the person to the previous ones. It calculates a velocity vector for every single pixel.
- **Mean Magnitude:** The intensities of all these vectors are averaged to create a single "Motion Magnitude" for that frame, representing how much the person is "shaking" or moving internally.

### 2. Rhythmic Jerk Detection (Frequency Analysis)
Once a 60-frame history is established, the `analyzeSeizure` method scans the motion signal:
- **Mean & Peaks:** It calculates the mean motion and looks for "peaks" (points where motion spikes 1.5x above the mean). 
- **Frequency Matching:** In an epileptic seizure, movements are rhythmic. The system counts these peaks. If the count corresponds to a frequency of **2Hz to 6Hz** (roughly 4 to 15 peaks in our 60-frame window), the person is flagged.
- **Noise Floor (`MIN_SEIZURE_MOTION`):** To prevent false alarms from camera sensor noise or compression artifacts, the mean motion must be above a threshold of **0.5**. If the "shaking" is too faint, it is ignored.

### 3. False Positive Filtering
To ensure reliability, the system employs two layers of filtering:
- **HOG hitThreshold (0.2):** The initial person detector is tuned to be "strict," requiring a high confidence score before it labels an object as a person.
- **Static Object Filter (`isLikelyStaticObject`):** If an object has been visible for more than 50 frames but its average lifetime motion is extremely low (**< 0.2**), the system assumes it is a stationary object (like a chair leg) and hides it from the UI.

## Tuning Person Detection (`detectMultiScale`)
The core detection logic in `VideoPanel.java` uses the OpenCV `hog.detectMultiScale` method. This method has several parameters that can be tuned to balance detection accuracy (sensitivity) vs. speed and false positives.

### Parameter Breakdown
```java
hog.detectMultiScale(resizedMat, detections, 0.2, new Size(8, 8), new Size(32, 32), 1.05, 2.0, false);
```

1.  **`hitThreshold` (0.2):** This is the most critical tuning parameter. It defines the threshold for the distance between the feature vector and the SVM classifying plane.
    *   **Increasing it (e.g., 0.2 to 0.5):** Makes the detector more "strict." It reduces false positives (like chair legs or shadows) but might miss people who are partially obscured or in low light.
    *   **Decreasing it (e.g., 0.2 to 0.0):** Makes it more "relaxed." It will catch every person but will significantly increase "ghost" detections of furniture and background patterns.
2.  **`winStride` (Size(8, 8)):** The "step size" of the sliding window. 
    *   Smaller values (e.g., 4x4) are more accurate but much slower.
    *   Larger values (e.g., 16x16) are faster but might "skip over" people.
3.  **`padding` (Size(32, 32)):** Adds a border around the search area to help detect people near the edges of the frame.
4.  **`scale` (1.05):** The factor by which the image is resized at each layer of the image pyramid. 
    *   **1.05** means a 5% reduction per layer. This is very thorough.
    *   **1.2** would be faster (fewer layers to search) but might miss people of specific sizes.
5.  **`groupThreshold` (2.0):** After searching, the algorithm finds many overlapping boxes. This parameter controls how they are grouped. A value of **2.0** requires at least two overlapping detections to confirm a person, which helps filter out random noise.

## Class Documentation

### `com.tapoviewer.TapoViewerMonitorApp`
The main entry point. It initializes the Swing event dispatch thread and launches the `MainFrame`.

### `com.tapoviewer.ui.MainFrame`
The primary window of the application. It manages the layout, housing the `VideoPanel` in the center and the `ControlPanel` on the right side. It also handles application shutdown by releasing resources.

### `com.tapoviewer.ui.VideoPanel`
Handles the video rendering logic using JavaCV.
- **FFmpeg Integration:** Uses `FFmpegFrameGrabber` to connect to the RTSP stream.
- **Threading:** Runs the frame-grabbing loop in a background daemon thread to keep the UI responsive.
- **Rendering:** Overrides `paintComponent` to draw the video frames onto the panel, maintaining the correct aspect ratio and scaling to fit the window.

### `com.tapoviewer.ui.ControlPanel`
The brain of the user interface.
- **Configuration Loading:** Reads `cameras.xml` to populate the House/Camera dropdowns and `secret.txt` for credentials.
- **State Management:** Tracks connection status and coordinates the handoff between ONVIF (control) and RTSP (video).
- **PTZ UI:** Manages the directional and zoom buttons, including the logic to enable/disable them based on camera capabilities.

### `com.tapoviewer.camera.CameraClient`
Orchestrates the ONVIF protocol.
- **Connection:** Authenticates with the camera and discovers service endpoints (Device, Media, and PTZ).
- **PTZ Execution:** Translates UI button clicks into ONVIF `ContinuousMove` commands.
- **Capability Detection:** Queries the camera's PTZ status to determine if hardware zoom is available.
- **Scheduler:** Uses a `ScheduledExecutorService` to handle "tap-to-move" behavior (sending a move command followed by a stop command after a 500ms delay).

### `com.tapoviewer.model.CameraSettings`
A simple data model that holds the camera's IP, ports, and credentials. It includes helper methods to construct the RTSP URL.

## Configuration Files

### `src/main/resources/cameras.xml`
Used to manage your camera list. Format:
```xml
<Cameras>
    <House name="My House">
        <Camera name="Living Room" ip="192.168.1.100"/>
    </House>
</Cameras>
```

### `src/main/resources/secret.txt`
Used to store your credentials securely. This file is excluded from Git via `.gitignore`.
```properties
username=your_camera_account_user
password=your_camera_account_password
```

## Prerequisites
1.  **Tapo Camera Account:** Ensure you have created a "Camera Account" in the Tapo App under **Settings > Advanced > Camera Account**.
2.  **Network:** Your camera and computer must be on the same network. For remote viewing, use a VPN (like Tailscale) or Port Forwarding.

## Running the Application
To run the application, use the provided Maven wrapper:

```bash
cd TapoViewerMonitor
export JAVA_HOME=/home/markvasey/.jdks/openjdk-23.0.1
export PATH=$JAVA_HOME/bin:$PATH
./mvnw clean compile exec:java -Dexec.mainClass="com.tapoviewer.TapoViewerMonitorApp"
  or, wit logging:
./mvnw clean compile exec:java -Dexec.mainClass="com.tapoviewer.TapoViewerMonitorApp" -Dslf4j.simpleLogger.defaultLogLevel=info
```

## Privacy & Data Handling
1.  **No Persistence Logic:** No credentials or IP addresses are ever saved to a database or sent to any external server.
2.  **Local Only:** All configuration files stay on your local machine.
3.  **In-Memory Only:** Credentials remain strictly in volatile memory while the app is active.

## References & Credits

- [HOG Feature Descriptor](https://medium.com/@dnemutlu/hog-feature-descriptor-263313c3b40d) - Dahi Nemutluw.

## License
MIT.
