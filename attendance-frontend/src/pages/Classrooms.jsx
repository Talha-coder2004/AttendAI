import { useState } from "react";

function Classrooms() {

    const [classrooms] = useState([
        {
            id: 1,
            name: "Room 1",
            building: "Main Block",
            capacity: 60,
            students: 1,
            cameras: 1,
            onlineCameras: 1,
            status: "ACTIVE"
        },
        {
            id: 2,
            name: "Room 2",
            building: "Main Block",
            capacity: 60,
            students: 0,
            cameras: 1,
            onlineCameras: 0,
            status: "MONITORING"
        }
    ]);

    return (
        <div className="page">

            <div className="page-header">

                <div>
                    <h1>Classrooms</h1>

                    <p>
                        Classroom occupancy and attendance monitoring
                    </p>
                </div>

                <div className="live-indicator">
                    <span></span>
                    LIVE
                </div>

            </div>


            {/* SUMMARY */}

            <div className="stats-grid">

                <div className="stat-card">

                    <div className="stat-icon blue">
                        ▣
                    </div>

                    <div className="stat-info">
                        <span>Total Classrooms</span>

                        <strong>
                            {classrooms.length}
                        </strong>

                        <small>
                            Registered classrooms
                        </small>
                    </div>

                </div>


                <div className="stat-card">

                    <div className="stat-icon green">
                        ✓
                    </div>

                    <div className="stat-info">
                        <span>Students Detected</span>

                        <strong>
                            {classrooms.reduce(
                                (sum, room) => sum + room.students,
                                0
                            )}
                        </strong>

                        <small className="green-text">
                            Currently present
                        </small>
                    </div>

                </div>


                <div className="stat-card">

                    <div className="stat-icon purple">
                        ◉
                    </div>

                    <div className="stat-info">
                        <span>Active Cameras</span>

                        <strong>
                            {classrooms.reduce(
                                (sum, room) => sum + room.onlineCameras,
                                0
                            )}
                        </strong>

                        <small>
                            Monitoring classrooms
                        </small>
                    </div>

                </div>

            </div>


            {/* CLASSROOM CARDS */}

            <div className="classroom-grid">

                {classrooms.map((room) => {

                    const occupancy =
                        room.capacity > 0
                            ? Math.round(
                                (room.students / room.capacity) * 100
                            )
                            : 0;

                    return (

                        <div
                            className="classroom-card"
                            key={room.id}
                        >

                            <div className="classroom-header">

                                <div>

                                    <div className="room-icon">
                                        🏫
                                    </div>

                                    <div>

                                        <h2>
                                            {room.name}
                                        </h2>

                                        <p>
                                            {room.building}
                                        </p>

                                    </div>

                                </div>


                                <span className="classroom-status">
                                    ● {room.status}
                                </span>

                            </div>


                            <div className="classroom-stats">

                                <div>
                                    <span>Students</span>

                                    <strong>
                                        {room.students}
                                    </strong>
                                </div>


                                <div>
                                    <span>Capacity</span>

                                    <strong>
                                        {room.capacity}
                                    </strong>
                                </div>


                                <div>
                                    <span>Cameras</span>

                                    <strong>
                                        {room.onlineCameras}/{room.cameras}
                                    </strong>
                                </div>

                            </div>


                            <div className="occupancy-section">

                                <div className="occupancy-title">

                                    <span>
                                        Occupancy
                                    </span>

                                    <strong>
                                        {occupancy}%
                                    </strong>

                                </div>


                                <div className="occupancy-bar">

                                    <div
                                        style={{
                                            width: `${occupancy}%`
                                        }}
                                    ></div>

                                </div>

                            </div>


                            <div className="classroom-footer">

                                <span>
                                    Camera monitoring
                                </span>

                                <span
                                    className={
                                        room.onlineCameras > 0
                                            ? "camera-connected"
                                            : "camera-disconnected"
                                    }
                                >
                                    {room.onlineCameras > 0
                                        ? "● Connected"
                                        : "● Offline"}
                                </span>

                            </div>

                        </div>

                    );

                })}

            </div>


            {/* CLASSROOM PROCESSING */}

            <div className="panel classroom-processing">

                <div className="panel-header">

                    <div>

                        <h2>
                            Classroom Presence Verification
                        </h2>

                        <p>
                            Continuous presence monitoring after identity
                            recognition
                        </p>

                    </div>

                </div>


                <div className="presence-flow">

                    <div className="presence-node">
                        📹
                        <strong>CCTV</strong>
                        <small>Classroom stream</small>
                    </div>

                    <div className="presence-arrow">
                        →
                    </div>

                    <div className="presence-node">
                        👤
                        <strong>Person Tracking</strong>
                        <small>Track students</small>
                    </div>

                    <div className="presence-arrow">
                        →
                    </div>

                    <div className="presence-node">
                        🧠
                        <strong>Re-Identification</strong>
                        <small>Verify presence</small>
                    </div>

                    <div className="presence-arrow">
                        →
                    </div>

                    <div className="presence-node">
                        ✓
                        <strong>Attendance</strong>
                        <small>Update status</small>
                    </div>

                </div>

            </div>

        </div>
    );
}

export default Classrooms;