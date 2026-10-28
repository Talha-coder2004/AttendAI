import { useState } from "react";

function Cameras() {

    const [cameras] = useState([
        {
            id: "CAM01",
            name: "Classroom Camera 01",
            classroom: "Room 1",
            location: "Main Classroom",
            status: "ONLINE",
            resolution: "1920 × 1080",
            fps: 25,
            people: 1
        },
        {
            id: "CAM02",
            name: "Classroom Camera 02",
            classroom: "Room 2",
            location: "Second Classroom",
            status: "OFFLINE",
            resolution: "1920 × 1080",
            fps: 0,
            people: 0
        }
    ]);

    const onlineCount =
        cameras.filter(camera => camera.status === "ONLINE").length;

    return (
        <div className="page">

            {/* HEADER */}

            <div className="page-header">

                <div>
                    <h1>Cameras</h1>

                    <p>
                        CCTV camera network and monitoring status
                    </p>
                </div>

                <div className="camera-summary">

                    <span className="online-dot"></span>

                    {onlineCount}/{cameras.length} cameras online

                </div>

            </div>


            {/* CAMERA SUMMARY */}

            <div className="stats-grid">

                <div className="stat-card">

                    <div className="stat-icon purple">
                        ◉
                    </div>

                    <div className="stat-info">

                        <span>Total Cameras</span>

                        <strong>
                            {cameras.length}
                        </strong>

                        <small>
                            Registered cameras
                        </small>

                    </div>

                </div>


                <div className="stat-card">

                    <div className="stat-icon green">
                        ✓
                    </div>

                    <div className="stat-info">

                        <span>Online</span>

                        <strong>
                            {onlineCount}
                        </strong>

                        <small className="green-text">
                            Monitoring active
                        </small>

                    </div>

                </div>


                <div className="stat-card">

                    <div className="stat-icon red">
                        !
                    </div>

                    <div className="stat-info">

                        <span>Offline</span>

                        <strong>
                            {cameras.length - onlineCount}
                        </strong>

                        <small>
                            Requires attention
                        </small>

                    </div>

                </div>

            </div>


            {/* CAMERA CARDS */}

            <div className="camera-grid">

                {cameras.map((camera) => (

                    <div
                        className="camera-management-card"
                        key={camera.id}
                    >

                        {/* PREVIEW */}

                        <div className="camera-preview">

                            <div className="preview-top">

                                <strong>
                                    {camera.id}
                                </strong>

                                <span
                                    className={
                                        camera.status === "ONLINE"
                                            ? "preview-online"
                                            : "preview-offline"
                                    }
                                >
                                    ● {camera.status}
                                </span>

                            </div>


                            {camera.status === "ONLINE" ? (

                                <>

                                    <div className="preview-grid"></div>

                                    <div className="preview-center">
                                        📹
                                    </div>

                                    <span className="tracking-label">
                                        AI MONITORING
                                    </span>

                                </>

                            ) : (

                                <div className="camera-offline-message">

                                    <span>⚠</span>

                                    <strong>
                                        Camera Offline
                                    </strong>

                                    <small>
                                        No video signal
                                    </small>

                                </div>

                            )}

                        </div>


                        {/* INFORMATION */}

                        <div className="camera-details">

                            <div>

                                <h2>
                                    {camera.name}
                                </h2>

                                <p>
                                    {camera.location}
                                </p>

                            </div>


                            <div className="camera-info-grid">

                                <div>

                                    <span>Classroom</span>

                                    <strong>
                                        {camera.classroom}
                                    </strong>

                                </div>


                                <div>

                                    <span>Resolution</span>

                                    <strong>
                                        {camera.resolution}
                                    </strong>

                                </div>


                                <div>

                                    <span>Frame Rate</span>

                                    <strong>
                                        {camera.fps > 0
                                            ? `${camera.fps} FPS`
                                            : "--"}
                                    </strong>

                                </div>


                                <div>

                                    <span>People Detected</span>

                                    <strong>
                                        {camera.people}
                                    </strong>

                                </div>

                            </div>


                            <div className="camera-actions">

                                <button
                                    className="secondary-button"
                                >
                                    View Monitor
                                </button>

                                <button
                                    className="secondary-button"
                                >
                                    Configure
                                </button>

                            </div>

                        </div>

                    </div>

                ))}

            </div>


            {/* MULTI CAMERA ARCHITECTURE */}

            <div className="panel">

                <div className="panel-header">

                    <div>

                        <h2>
                            Multi-Camera Processing
                        </h2>

                        <p>
                            AI processing architecture for multiple
                            classroom streams
                        </p>

                    </div>

                </div>


                <div className="camera-architecture">

                    <div className="architecture-node">
                        📹
                        <strong>CAM01</strong>
                        <small>Room 1</small>
                    </div>

                    <div className="architecture-arrow">
                        →
                    </div>

                    <div className="architecture-node">
                        🧠
                        <strong>AI Node</strong>
                        <small>Detection + Tracking</small>
                    </div>

                    <div className="architecture-arrow">
                        →
                    </div>

                    <div className="architecture-node">
                        ⚡
                        <strong>Fusion</strong>
                        <small>Identity Decision</small>
                    </div>

                    <div className="architecture-arrow">
                        →
                    </div>

                    <div className="architecture-node">
                        ☕
                        <strong>Backend</strong>
                        <small>Attendance Event</small>
                    </div>

                </div>

            </div>

        </div>
    );
}

export default Cameras;