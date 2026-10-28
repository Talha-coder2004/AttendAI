import { useEffect, useState } from "react";
import { getStudents } from "../services/api";

function Students() {

    const [students, setStudents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    const loadStudents = async () => {

        try {

            setLoading(true);
            setError("");

            const response = await getStudents();

            setStudents(response.data);

        } catch (err) {

            console.error(err);
            setError("Unable to load students.");

        } finally {

            setLoading(false);

        }
    };

    useEffect(() => {
        loadStudents();
    }, []);

    return (
        <div className="page">

            <div className="page-header">

                <div>
                    <h1>Students</h1>

                    <p>
                        Registered students and biometric enrollment
                    </p>
                </div>

                <button
                    className="refresh-button"
                    onClick={loadStudents}
                >
                    ↻ Refresh
                </button>

            </div>


            {error && (
                <div className="error-banner">
                    ⚠ {error}
                </div>
            )}


            <div className="panel">

                <div className="panel-header">

                    <div>
                        <h2>Registered Students</h2>

                        <p>
                            Students currently registered in the system
                        </p>
                    </div>

                    <div className="record-count">
                        {students.length} students
                    </div>

                </div>


                {loading ? (

                    <div className="page-loading">
                        Loading students...
                    </div>

                ) : students.length === 0 ? (

                    <div className="page-empty">
                        <strong>No students registered</strong>
                        <span>
                            Students will appear here after enrollment.
                        </span>
                    </div>

                ) : (

                    <div className="table-container">

                        <table>

                            <thead>

                                <tr>
                                    <th>STUDENT ID</th>
                                    <th>NAME</th>
                                    <th>REGISTER NUMBER</th>
                                    <th>DEPARTMENT</th>
                                    <th>CLASSROOM</th>
                                    <th>STATUS</th>
                                </tr>

                            </thead>

                            <tbody>

                                {students.map((student) => (

                                    <tr key={student.id}>

                                        <td>
                                            <strong>
                                                {student.studentId}
                                            </strong>
                                        </td>

                                        <td>
                                            {student.name}
                                        </td>

                                        <td>
                                            {student.registerNumber}
                                        </td>

                                        <td>
                                            {student.department}
                                        </td>

                                        <td>
                                            Room {student.classroomId}
                                        </td>

                                        <td>

                                            <span className="status-badge">
                                                <span></span>
                                                {student.status}
                                            </span>

                                        </td>

                                    </tr>

                                ))}

                            </tbody>

                        </table>

                    </div>

                )}

            </div>

        </div>
    );
}

export default Students;