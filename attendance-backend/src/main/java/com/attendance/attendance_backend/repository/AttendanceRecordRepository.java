package com.attendance.attendance_backend.repository;

import com.attendance.attendance_backend.entity.AttendanceRecord;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface AttendanceRecordRepository
        extends JpaRepository<AttendanceRecord, Long> {

    Optional<AttendanceRecord> findByStudentIdAndClassroomIdAndAttendanceDate(
            String studentId,
            Long classroomId,
            LocalDate attendanceDate
    );

    List<AttendanceRecord> findByAttendanceDate(LocalDate date);

    List<AttendanceRecord> findByClassroomIdAndAttendanceDate(
            Long classroomId,
            LocalDate date
    );
}