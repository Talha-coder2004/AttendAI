import { useEffect, useState } from "react";

function LiveMonitoring() {

    const [cameraOnline, setCameraOnline] = useState(true);

    const [events, setEvents] = useState([
        {
            id: 1,
            studentId: "STU001",
            classroom: "Room 1",
            direction: "IN",
            face: 94,
            gait: 81,
            body: 87,
            fusion: 93,
            time: "08:00 PM"
        }
    ]);


    // Simulate camera heartbeat
    useEffect(() => {

        const interval = setInterval(() => {

            setCameraOnline(true);

        }, 5000);

        return () => clearInterval(interval);

    }, []);


    return (

        <div className="page">

            {/* HEADER */}

            <div className="page-header">

                <div>

                    <h1>
                        Live Monitoring
                    </h1>

                    <p>
                        Real-time CCTV and multimodal AI
                        recognition
                    </p>

                </div>


                <div className="live-indicator">

                    <span></span>

                    LIVE

                </div>

            </div>


            {/* CAMERA + RECOGNITION */}

            <div className="monitor-grid">


                {/* CAMERA */}

                <div className="camera-view">

                    <div className="camera-header">

                        <div>

                            <strong>
                                CAM01
                            </strong>

                            <small>
                                Classroom 1
                            </small>

                        </div>


                        <span className="camera-live">

                            {cameraOnline
                                ? "ONLINE"
                                : "OFFLINE"}

                        </span>

                    </div>


                    <div className="camera-placeholder">

                        <div className="camera-overlay">

                            <div className="detection-box">

                                <span className="box-label">
                                    TRACK-001
                                </span>

                            </div>

                        </div>


                        <div className="camera-message">

                            <div className="camera-icon">
                                📹
                            </div>

                            <strong>
                                CCTV Camera Feed
                            </strong>

                            <small>
                                AI processing stream
                            </small>

                        </div>

                    </div>

                </div>


                {/* RECOGNITION PANEL */}

                <div className="panel recognition-panel">

                    <div className="panel-header">

                        <div>

                            <h2>
                                AI Recognition
                            </h2>

                            <p>
                                Latest identity decision
                            </p>

                        </div>


                        <span className="status-badge">

                            <span></span>

                            IDENTIFIED

                        </span>

                    </div>


                    {events.length > 0 && (

                        <div className="recognition-content">


                            {/* STUDENT */}

                            <div className="recognized-student">

                                <div className="large-avatar">
                                    ST
                                </div>

                                <div>

                                    <strong>
                                        {events[0].studentId}
                                    </strong>

                                    <span>
                                        Student • Room 1
                                    </span>

                                </div>

                            </div>


                            {/* MODALITY SCORES */}

                            <div className="modality-section">

                                <h3>
                                    Multimodal Evidence
                                </h3>


                                <div className="modality-row">

                                    <span>
                                        Face Recognition
                                    </span>

                                    <div className="modality-bar">

                                        <div
                                            style={{
                                                width:
                                                    `${events[0].face}%`
                                            }}
                                        />

                                    </div>

                                    <strong>
                                        {events[0].face}%
                                    </strong>

                                </div>


                                <div className="modality-row">

                                    <span>
                                        Gait Recognition
                                    </span>

                                    <div className="modality-bar">

                                        <div
                                            style={{
                                                width:
                                                    `${events[0].gait}%`
                                            }}
                                        />

                                    </div>

                                    <strong>
                                        {events[0].gait}%
                                    </strong>

                                </div>


                                <div className="modality-row">

                                    <span>
                                        Body Re-ID
                                    </span>

                                    <div className="modality-bar">

                                        <div
                                            style={{
                                                width:
                                                    `${events[0].body}%`
                                            }}
                                        />

                                    </div>

                                    <strong>
                                        {events[0].body}%
                                    </strong>

                                </div>

                            </div>


                            {/* FUSION */}

                            <div className="fusion-box">

                                <div>

                                    <span>
                                        Adaptive Fusion
                                    </span>

                                    <small>
                                        Combined confidence
                                    </small>

                                </div>


                                <strong>
                                    {events[0].fusion}%
                                </strong>

                            </div>


                            {/* DECISION */}

                            <div className="decision-row">

                                <div>

                                    <span>
                                        Final Decision
                                    </span>

                                    <strong>
                                        {events[0].studentId}
                                    </strong>

                                </div>


                                <div className="direction in">
                                    ↓ {events[0].direction}
                                </div>

                            </div>

                        </div>

                    )}

                </div>

            </div>


            {/* PIPELINE */}

            <div className="panel monitoring-pipeline">

                <div className="panel-header">

                    <div>

                        <h2>
                            AI Processing Pipeline
                        </h2>

                        <p>
                            Current recognition flow
                        </p>

                    </div>

                </div>


                <div className="pipeline">


                    <div className="pipeline-step active">

                        <div>📹</div>

                        <span>
                            CCTV
                        </span>

                    </div>


                    <div className="pipeline-line active"></div>


                    <div className="pipeline-step active">

                        <div>👁</div>

                        <span>
                            Detection
                        </span>

                    </div>


                    <div className="pipeline-line active"></div>


                    <div className="pipeline-step active">

                        <div>🎯</div>

                        <span>
                            Tracking
                        </span>

                    </div>


                    <div className="pipeline-line active"></div>


                    <div className="pipeline-step active">

                        <div>🧠</div>

                        <span>
                            Face / Gait / Body
                        </span>

                    </div>


                    <div className="pipeline-line active"></div>


                    <div className="pipeline-step active">

                        <div>⚡</div>

                        <span>
                            Fusion
                        </span>

                    </div>


                    <div className="pipeline-line active"></div>


                    <div className="pipeline-step active">

                        <div>✓</div>

                        <span>
                            Attendance
                        </span>

                    </div>

                </div>

            </div>


            {/* RECENT EVENTS */}

            <div className="panel">

                <div className="panel-header">

                    <div>

                        <h2>
                            Recent Recognition Events
                        </h2>

                        <p>
                            Latest AI identity decisions
                        </p>

                    </div>

                    <div className="record-count">
                        {events.length} events
                    </div>

                </div>


                <div className="table-container">

                    <table>

                        <thead>

                            <tr>

                                <th>
                                    TRACK
                                </th>

                                <th>
                                    STUDENT
                                </th>

                                <th>
                                    CAMERA
                                </th>

                                <th>
                                    DIRECTION
                                </th>

                                <th>
                                    CONFIDENCE
                                </th>

                                <th>
                                    TIME
                                </th>

                            </tr>

                        </thead>


                        <tbody>

                            {events.map((event) => (

                                <tr key={event.id}>

                                    <td>
                                        TRACK-{String(event.id).padStart(3, "0")}
                                    </td>

                                    <td>
                                        <strong>
                                            {event.studentId}
                                        </strong>
                                    </td>

                                    <td>
                                        CAM01
                                    </td>

                                    <td>

                                        <span className="direction in">
                                            ↓ {event.direction}
                                        </span>

                                    </td>

                                    <td>

                                        <div className="confidence">

                                            <div className="confidence-bar">

                                                <div
                                                    style={{
                                                        width:
                                                            `${event.fusion}%`
                                                    }}
                                                />

                                            </div>

                                            <span>
                                                {event.fusion}%
                                            </span>

                                        </div>

                                    </td>

                                    <td>
                                        {event.time}
                                    </td>

                                </tr>

                            ))}

                        </tbody>

                    </table>

                </div>

            </div>

        </div>
    );
}

export default LiveMonitoring;