package com.attendance.attendance_backend.repository;

import com.attendance.attendance_backend.entity.AttendanceEvent;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDateTime;
import java.util.List;

public interface AttendanceEventRepository
        extends JpaRepository<AttendanceEvent, Long> {

    List<AttendanceEvent> findByStudentId(String studentId);

    List<AttendanceEvent> findByEventTimeBetween(
            LocalDateTime start,
            LocalDateTime end
    );
}