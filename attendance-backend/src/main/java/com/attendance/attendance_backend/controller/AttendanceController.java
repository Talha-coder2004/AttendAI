package com.attendance.attendance_backend.controller;

import com.attendance.attendance_backend.entity.AttendanceRecord;
import com.attendance.attendance_backend.service.AttendanceService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/attendance")
public class AttendanceController {

    private final AttendanceService attendanceService;

    public AttendanceController(
            AttendanceService attendanceService) {

        this.attendanceService = attendanceService;
    }

    @GetMapping("/today")
    public ResponseEntity<List<AttendanceRecord>> getTodayAttendance() {
        return ResponseEntity.ok(
                attendanceService.getTodayAttendance()
        );
    }

    @GetMapping("/classroom/{classroomId}")
    public ResponseEntity<List<AttendanceRecord>>
    getClassroomAttendance(
            @PathVariable Long classroomId) {

        return ResponseEntity.ok(
                attendanceService
                        .getClassroomAttendance(classroomId)
        );
    }
}