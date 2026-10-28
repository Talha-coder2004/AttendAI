function Reports() {

    return (
        <div className="page">

            <div className="page-header">

                <div>

                    <h1>Reports</h1>

                    <p>
                        Attendance reports and analytics
                    </p>

                </div>

            </div>


            <div className="stats-grid">

                <div className="stat-card">
                    <div className="stat-info">
                        <span>Daily Attendance</span>
                        <strong>--</strong>
                        <small>
                            Today's percentage
                        </small>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-info">
                        <span>Weekly Attendance</span>
                        <strong>--</strong>
                        <small>
                            Current week
                        </small>
                    </div>
                </div>

                <div className="stat-card">
                    <div className="stat-info">
                        <span>Present Students</span>
                        <strong>--</strong>
                        <small>
                            Today's total
                        </small>
                    </div>
                </div>

            </div>


            <div className="panel">

                <div className="page-empty">

                    <strong>
                        Attendance reports
                    </strong>

                    <span>
                        Detailed reports will be connected
                        to the attendance API next.
                    </span>

                </div>

            </div>

        </div>
    );
}

export default Reports;