package com.attendance.attendance_backend.service;

import com.attendance.attendance_backend.dto.DashboardSummary;
import com.attendance.attendance_backend.repository.AttendanceRecordRepository;
import com.attendance.attendance_backend.repository.CameraRepository;
import com.attendance.attendance_backend.repository.StudentRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDate;

@Service
public class DashboardService {

    private final StudentRepository studentRepository;
    private final CameraRepository cameraRepository;
    private final AttendanceRecordRepository attendanceRepository;

    public DashboardService(
            StudentRepository studentRepository,
            CameraRepository cameraRepository,
            AttendanceRecordRepository attendanceRepository) {

        this.studentRepository = studentRepository;
        this.cameraRepository = cameraRepository;
        this.attendanceRepository = attendanceRepository;
    }

    public DashboardSummary getSummary() {

        long totalStudents = studentRepository.countByStatus("ACTIVE");

        long presentStudents =
                attendanceRepository
                        .findByAttendanceDate(LocalDate.now())
                        .size();

        long absentStudents =
                Math.max(0, totalStudents - presentStudents);

        long totalCameras = cameraRepository.count();

        long onlineCameras =
                cameraRepository.countByStatus("ONLINE");

        return new DashboardSummary(
                totalStudents,
                presentStudents,
                absentStudents,
                totalCameras,
                onlineCameras
        );
    }
}