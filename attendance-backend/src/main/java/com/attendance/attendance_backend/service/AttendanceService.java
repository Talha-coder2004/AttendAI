package com.attendance.attendance_backend.service;

import com.attendance.attendance_backend.entity.AttendanceEvent;
import com.attendance.attendance_backend.entity.AttendanceRecord;
import com.attendance.attendance_backend.repository.AttendanceRecordRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

@Service
public class AttendanceService {

    private final AttendanceRecordRepository attendanceRepository;

    public AttendanceService(
            AttendanceRecordRepository attendanceRepository) {

        this.attendanceRepository = attendanceRepository;
    }

    public AttendanceRecord processAttendance(
            AttendanceEvent event) {

        // Only confirmed identity events can mark attendance
        if (!"IDENTITY_CONFIRMED".equals(event.getEventType())) {
            return null;
        }

        // Only valid IN events mark presence
        if (!"IN".equalsIgnoreCase(event.getDirection())) {
            return null;
        }

        if (event.getConfidence() == null ||
                event.getConfidence() < 0.70) {
            return null;
        }

        LocalDate date = event.getEventTime().toLocalDate();

        // Check whether attendance already exists
        var existing = attendanceRepository
                .findByStudentIdAndClassroomIdAndAttendanceDate(
                        event.getStudentId(),
                        event.getClassroomId(),
                        date
                );

        if (existing.isPresent()) {
            return existing.get();
        }

        AttendanceRecord record = new AttendanceRecord();

        record.setStudentId(event.getStudentId());
        record.setClassroomId(event.getClassroomId());
        record.setAttendanceDate(date);
        record.setStatus("PRESENT");
        record.setMarkedAt(
                event.getEventTime() != null
                        ? event.getEventTime()
                        : LocalDateTime.now()
        );
        record.setConfidence(event.getConfidence());

        return attendanceRepository.save(record);
    }

    public List<AttendanceRecord> getTodayAttendance() {
        return attendanceRepository.findByAttendanceDate(
                LocalDate.now()
        );
    }

    public List<AttendanceRecord> getClassroomAttendance(
            Long classroomId) {

        return attendanceRepository
                .findByClassroomIdAndAttendanceDate(
                        classroomId,
                        LocalDate.now()
                );
    }
}