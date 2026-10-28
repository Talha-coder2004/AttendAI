package com.attendance.attendance_backend.controller;

import com.attendance.attendance_backend.dto.IdentityEventRequest;
import com.attendance.attendance_backend.entity.AttendanceEvent;
import com.attendance.attendance_backend.service.AttendanceEventService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/events")
public class AttendanceEventController {

    private final AttendanceEventService eventService;

    public AttendanceEventController(
            AttendanceEventService eventService) {

        this.eventService = eventService;
    }

    @PostMapping("/identity")
    public ResponseEntity<AttendanceEvent> receiveIdentityEvent(
            @RequestBody IdentityEventRequest request) {

        AttendanceEvent event =
                eventService.processIdentityEvent(request);

        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(event);
    }

    @GetMapping
    public ResponseEntity<List<AttendanceEvent>> getAllEvents() {
        return ResponseEntity.ok(eventService.getAllEvents());
    }

    @GetMapping("/student/{studentId}")
    public ResponseEntity<List<AttendanceEvent>> getStudentEvents(
            @PathVariable String studentId) {

        return ResponseEntity.ok(
                eventService.getStudentEvents(studentId)
        );
    }
}