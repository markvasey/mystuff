# TapoViewer

A Java Swing application to view Tapo cameras via RTSP and control them via ONVIF.

## Key Features
- **RTSP Video Streaming:** Powered by **JavaCV (FFmpeg)**. This provides high-performance, low-latency streaming and is guaranteed to work if `ffplay` works on your system.
- **ONVIF PTZ Control:** Uses the modern, asynchronous `onvif-java` library to provide Pan, Tilt, and Zoom functionality.
- **Dynamic Camera Selection:** Organise cameras by "House" and "Name" via an XML configuration file.
- **Automatic Switching:** The app automatically disconnects from the current camera and connects to the new one when you change the dropdown selection.
- **Hardware Detection:** Automatically detects if a camera supports hardware zoom and enables/disables the UI buttons accordingly.
- **Modern Java:** Built for Java 17+ (verified with OpenJDK 23.0.1 environment).
- **Self-Contained:** No external media players (like VLC) are required as JavaCV bundles its own FFmpeg binaries.

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
cd TapoViewer
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

## License
MIT.
