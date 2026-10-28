package com.attendance.attendance_backend.dto;

public class DashboardSummary {

    private long totalStudents;
    private long presentStudents;
    private long absentStudents;
    private long totalCameras;
    private long onlineCameras;

    public DashboardSummary() {
    }

    public DashboardSummary(
            long totalStudents,
            long presentStudents,
            long absentStudents,
            long totalCameras,
            long onlineCameras) {

        this.totalStudents = totalStudents;
        this.presentStudents = presentStudents;
        this.absentStudents = absentStudents;
        this.totalCameras = totalCameras;
        this.onlineCameras = onlineCameras;
    }

    public long getTotalStudents() {
        return totalStudents;
    }

    public long getPresentStudents() {
        return presentStudents;
    }

    public long getAbsentStudents() {
        return absentStudents;
    }

    public long getTotalCameras() {
        return totalCameras;
    }

    public long getOnlineCameras() {
        return onlineCameras;
    }
}