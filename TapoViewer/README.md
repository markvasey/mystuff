# TapoViewer

A Java Swing application to view Tapo cameras via RTSP and control them via ONVIF.

## Key Features
- **RTSP Video Streaming:** Powered by **JavaCV (FFmpeg)**. This provides high-performance, low-latency streaming and is guaranteed to work if `ffplay` works on your system.
- **ONVIF PTZ Control:** Uses the modern, asynchronous `onvif-java` library to provide Pan, Tilt, and Zoom functionality.
- **Modern Java:** Built for Java 17+ (verified with OpenJDK 23.0.1 environment).
- **Self-Contained:** No external media players (like VLC) are required.

## Project Structure
- `com.tapoviewer.TapoViewerApp`: Main entry point.
- `com.tapoviewer.ui`: Contains the Swing GUI components.
- `com.tapoviewer.camera`: Handles the logic for RTSP and ONVIF connections.
- `com.tapoviewer.model`: Data models for camera settings.

## Prerequisites
1.  **Tapo Camera Account:** Ensure you have created a "Camera Account" in the Tapo App under **Settings > Advanced > Camera Account**.
2.  **Network:** Your camera and computer must be on the same network.

## Running the Application
To run the application, use the provided Maven wrapper:

```bash
cd TapoViewer
export JAVA_HOME=/home/markvasey/.jdks/openjdk-23.0.1
export PATH=$JAVA_HOME/bin:$PATH
./mvnw clean compile exec:java -Dexec.mainClass="com.tapoviewer.TapoViewerApp"
```

## Privacy & Data Handling
1.  **No Persistence Logic:** No credentials are ever saved to disk.
2.  **In-Memory Only:** All credentials remain strictly in volatile memory.
3.  **Logging Safety:** Usernames and passwords are never logged.

## License
MIT.
