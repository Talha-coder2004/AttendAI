import { useEffect, useState } from "react";

import {
    getDashboardSummary,
    getTodayAttendance
} from "../services/api";


function Dashboard() {

    const [summary, setSummary] = useState({
        totalStudents: 0,
        presentStudents: 0,
        absentStudents: 0,
        totalCameras: 0,
        onlineCameras: 0
    });

    const [attendance, setAttendance] = useState([]);

    const [loading, setLoading] = useState(true);

    const [error, setError] = useState("");


    // =========================
    // LOAD DASHBOARD DATA
    // =========================

    const loadDashboard = async () => {

        try {

            setError("");

            const [
                summaryResponse,
                attendanceResponse
            ] = await Promise.all([

                getDashboardSummary(),

                getTodayAttendance()

            ]);


            setSummary(summaryResponse.data);

            setAttendance(attendanceResponse.data.value || []);


        } catch (err) {

            console.error(
                "Dashboard API error:",
                err
            );

            setError(
                "Unable to connect to the attendance backend."
            );

        } finally {

            setLoading(false);

        }

    };


    // =========================
    // INITIAL LOAD + REFRESH
    // =========================

    useEffect(() => {

        loadDashboard();


        // Temporary polling.
        // Later this will be replaced
        // with WebSocket live events.

        const interval = setInterval(
            loadDashboard,
            5000
        );


        return () => {

            clearInterval(interval);

        };

    }, []);


    // =========================
    // ATTENDANCE %
    // =========================

    const attendancePercentage =
        summary.totalStudents > 0

            ? Math.round(
                (
                    summary.presentStudents /
                    summary.totalStudents
                ) * 100
            )

            : 0;


    // Circle angle

    const circleAngle =
        attendancePercentage * 3.6;


    // =========================
    // LOADING SCREEN
    // =========================

    if (loading) {

        return (

            <div className="loading-screen">

                <div className="loader"></div>

                <h2>
                    Loading Attendance System
                </h2>

                <p>
                    Connecting to backend...
                </p>

            </div>

        );

    }


    // =========================
    // DASHBOARD
    // =========================

    return (

        <>

            {/* =========================
                TOP HEADER
            ========================== */}

            <header className="topbar">

                <div>

                    <h1>
                        Attendance Dashboard
                    </h1>

                    <p>
                        AI-powered real-time classroom
                        attendance monitoring
                    </p>

                </div>


                <div className="topbar-right">

                    <div className="live-indicator">

                        <span></span>

                        LIVE

                    </div>


                    <button
                        className="refresh-button"
                        onClick={loadDashboard}
                    >

                        ↻ Refresh

                    </button>

                </div>

            </header>


            {/* =========================
                ERROR
            ========================== */}

            {error && (

                <div className="error-banner">

                    ⚠ {error}

                </div>

            )}


            {/* =========================
                SUMMARY CARDS
            ========================== */}

            <section className="stats-grid">


                {/* TOTAL STUDENTS */}

                <div className="stat-card">

                    <div className="stat-icon blue">
                        👥
                    </div>


                    <div className="stat-info">

                        <span>
                            Total Students
                        </span>

                        <strong>
                            {summary.totalStudents}
                        </strong>

                        <small>
                            Registered students
                        </small>

                    </div>

                </div>


                {/* PRESENT */}

                <div className="stat-card">

                    <div className="stat-icon green">
                        ✓
                    </div>


                    <div className="stat-info">

                        <span>
                            Present
                        </span>

                        <strong>
                            {summary.presentStudents}
                        </strong>

                        <small className="green-text">

                            {attendancePercentage}%
                            attendance

                        </small>

                    </div>

                </div>


                {/* ABSENT */}

                <div className="stat-card">

                    <div className="stat-icon red">
                        !
                    </div>


                    <div className="stat-info">

                        <span>
                            Absent
                        </span>

                        <strong>
                            {summary.absentStudents}
                        </strong>

                        <small>
                            Today's attendance
                        </small>

                    </div>

                </div>


                {/* CAMERAS */}

                <div className="stat-card">

                    <div className="stat-icon purple">
                        ◉
                    </div>


                    <div className="stat-info">

                        <span>
                            Cameras Online
                        </span>


                        <strong>

                            {summary.onlineCameras}

                            <em>
                                / {summary.totalCameras}
                            </em>

                        </strong>


                        <small className="green-text">

                            Monitoring active

                        </small>

                    </div>

                </div>

            </section>


            {/* =========================
                MAIN CONTENT
            ========================== */}

            <section className="content-grid">


                {/* =========================
                    ATTENDANCE TABLE
                ========================== */}

                <div className="panel attendance-panel">


                    <div className="panel-header">

                        <div>

                            <h2>
                                Today's Attendance
                            </h2>

                            <p>
                                Recently recognized students
                            </p>

                        </div>


                        <div className="record-count">

                            {attendance.length}

                            {" "}

                            {attendance.length === 1
                                ? "record"
                                : "records"}

                        </div>

                    </div>


                    <div className="table-container">

                        <table>

                            <thead>

                                <tr>

                                    <th>
                                        STUDENT
                                    </th>

                                    <th>
                                        CLASSROOM
                                    </th>

                                    <th>
                                        STATUS
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


                                {attendance.length === 0 ? (

                                    <tr>

                                        <td
                                            colSpan="5"
                                            className="empty-state"
                                        >

                                            <div>

                                                <span>
                                                    ◌
                                                </span>

                                                <strong>
                                                    No attendance records
                                                </strong>

                                                <small>
                                                    Waiting for AI
                                                    recognition events
                                                </small>

                                            </div>

                                        </td>

                                    </tr>

                                ) : (


                                    attendance.map(
                                        (record) => (

                                            <tr
                                                key={
                                                    record.id
                                                }
                                            >


                                                {/* STUDENT */}

                                                <td>

                                                    <div className="student-cell">


                                                        <div className="avatar">

                                                            {record.studentId
                                                                ? record.studentId
                                                                    .substring(0, 2)
                                                                    .toUpperCase()

                                                                : "ST"}

                                                        </div>


                                                        <div>

                                                            <strong>

                                                                {
                                                                    record.studentId ||
                                                                    "Unknown"
                                                                }

                                                            </strong>


                                                            <span>
                                                                Student
                                                            </span>

                                                        </div>


                                                    </div>

                                                </td>


                                                {/* CLASSROOM */}

                                                <td>

                                                    Room{" "}

                                                    {
                                                        record.classroomId ??
                                                        "-"
                                                    }

                                                </td>


                                                {/* STATUS */}

                                                <td>

                                                    <span className="status-badge">

                                                        <span></span>

                                                        {
                                                            record.status ||
                                                            "PRESENT"
                                                        }

                                                    </span>

                                                </td>


                                                {/* CONFIDENCE */}

                                                <td>

                                                    <div className="confidence">


                                                        <div className="confidence-bar">

                                                            <div
                                                                style={{
                                                                    width:
                                                                        `${(
                                                                            (record.confidence || 0)
                                                                            * 100
                                                                        )}%`
                                                                }}
                                                            ></div>

                                                        </div>


                                                        <span>

                                                            {record.confidence != null

                                                                ? `${(
                                                                    record.confidence *
                                                                    100
                                                                ).toFixed(1)}%`

                                                                : "-"

                                                            }

                                                        </span>

                                                    </div>

                                                </td>


                                                {/* TIME */}

                                                <td>

                                                    {record.markedAt

                                                        ? new Date(
                                                            record.markedAt
                                                        ).toLocaleTimeString(
                                                            [],
                                                            {
                                                                hour:
                                                                    "2-digit",

                                                                minute:
                                                                    "2-digit"
                                                            }
                                                        )

                                                        : record.eventTime

                                                            ? new Date(
                                                                record.eventTime
                                                            ).toLocaleTimeString(
                                                                [],
                                                                {
                                                                    hour:
                                                                        "2-digit",

                                                                    minute:
                                                                        "2-digit"
                                                                }
                                                            )

                                                            : "-"

                                                    }

                                                </td>


                                            </tr>

                                        )
                                    )

                                )}

                            </tbody>

                        </table>

                    </div>

                </div>


                {/* =========================
                    RIGHT COLUMN
                ========================== */}

                <div className="right-column">


                    {/* =========================
                        ATTENDANCE OVERVIEW
                    ========================== */}

                    <div className="panel overview-panel">


                        <div className="panel-header">

                            <div>

                                <h2>
                                    Today's Overview
                                </h2>

                                <p>
                                    Attendance distribution
                                </p>

                            </div>

                        </div>


                        <div
                            className="attendance-circle"

                            style={{
                                "--progress":
                                    `${circleAngle}deg`
                            }}

                        >

                            <div className="circle-inner">

                                <strong>
                                    {attendancePercentage}%
                                </strong>

                                <span>
                                    Present
                                </span>

                            </div>

                        </div>


                        <div className="overview-legend">


                            <div>

                                <span
                                    className="legend-dot present-dot"
                                ></span>

                                Present

                                <strong>
                                    {summary.presentStudents}
                                </strong>

                            </div>


                            <div>

                                <span
                                    className="legend-dot absent-dot"
                                ></span>

                                Absent

                                <strong>
                                    {summary.absentStudents}
                                </strong>

                            </div>


                        </div>

                    </div>


                    {/* =========================
                        CAMERA NETWORK
                    ========================== */}

                    <div className="panel camera-panel">


                        <div className="panel-header">

                            <div>

                                <h2>
                                    Camera Network
                                </h2>

                                <p>
                                    CCTV monitoring status
                                </p>

                            </div>


                            <span className="camera-live">

                                LIVE

                            </span>

                        </div>


                        <div className="camera-status-row">


                            {/* ONLINE */}

                            <div className="camera-number">

                                <strong>

                                    {summary.onlineCameras}

                                </strong>

                                <span>
                                    Online
                                </span>

                            </div>


                            {/* OFFLINE */}

                            <div className="camera-number offline">

                                <strong>

                                    {Math.max(
                                        0,
                                        summary.totalCameras -
                                        summary.onlineCameras
                                    )}

                                </strong>

                                <span>
                                    Offline
                                </span>

                            </div>


                        </div>


                        <div className="camera-progress">

                            <div
                                style={{
                                    width:
                                        summary.totalCameras > 0

                                            ? `${(
                                                summary.onlineCameras /
                                                summary.totalCameras
                                            ) * 100}%`

                                            : "0%"
                                }}
                            ></div>

                        </div>


                        <small>

                            {summary.totalCameras > 0

                                ? `${Math.round(
                                    (
                                        summary.onlineCameras /
                                        summary.totalCameras
                                    ) * 100
                                )}% of cameras operational`

                                : "No cameras registered"

                            }

                        </small>

                    </div>


                </div>

            </section>


            {/* =========================
                AI PIPELINE
            ========================== */}

            <section className="panel architecture-panel">


                <div className="panel-header">

                    <div>

                        <h2>
                            AI Attendance Pipeline
                        </h2>

                        <p>
                            Current system architecture
                        </p>

                    </div>

                </div>


                <div className="pipeline">


                    <div className="pipeline-step">

                        <div>
                            📹
                        </div>

                        <span>
                            CCTV
                        </span>

                    </div>


                    <div className="pipeline-line"></div>


                    <div className="pipeline-step">

                        <div>
                            👁
                        </div>

                        <span>
                            Detection
                        </span>

                    </div>


                    <div className="pipeline-line"></div>


                    <div className="pipeline-step">

                        <div>
                            🧠
                        </div>

                        <span>
                            Multimodal AI
                        </span>

                    </div>


                    <div className="pipeline-line"></div>


                    <div className="pipeline-step">

                        <div>
                            ⚡
                        </div>

                        <span>
                            Fusion
                        </span>

                    </div>


                    <div className="pipeline-line"></div>


                    <div className="pipeline-step">

                        <div>
                            ☕
                        </div>

                        <span>
                            Spring Boot
                        </span>

                    </div>


                    <div className="pipeline-line"></div>


                    <div className="pipeline-step">

                        <div>
                            🗄
                        </div>

                        <span>
                            PostgreSQL
                        </span>

                    </div>


                </div>

            </section>


            {/* =========================
                FOOTER
            ========================== */}

            <footer className="footer">

                <span>
                    AI-Based Multimodal Video Attendance System
                </span>


                <span>
                    Backend: Spring Boot •
                    Database: PostgreSQL •
                    Frontend: React
                </span>

            </footer>

        </>

    );

}


export default Dashboard;