package com.attendance.attendance_backend.service;

import com.attendance.attendance_backend.dto.IdentityEventRequest;
import com.attendance.attendance_backend.entity.AttendanceEvent;
import com.attendance.attendance_backend.entity.Camera;
import com.attendance.attendance_backend.repository.AttendanceEventRepository;
import com.attendance.attendance_backend.repository.CameraRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class AttendanceEventService {

    private final AttendanceEventRepository eventRepository;
    private final CameraRepository cameraRepository;
    private final AttendanceService attendanceService;

    public AttendanceEventService(
            AttendanceEventRepository eventRepository,
            CameraRepository cameraRepository,
            AttendanceService attendanceService) {

        this.eventRepository = eventRepository;
        this.cameraRepository = cameraRepository;
        this.attendanceService = attendanceService;
    }

    public AttendanceEvent processIdentityEvent(
            IdentityEventRequest request) {

        Camera camera = cameraRepository
                .findByCameraId(request.getCameraId())
                .orElseThrow(() ->
                        new RuntimeException("Camera not found"));

        AttendanceEvent event = new AttendanceEvent();

        event.setStudentId(request.getCandidateId());
        event.setCameraId(request.getCameraId());
        event.setClassroomId(camera.getClassroomId());
        event.setEventType(request.getEventType());
        event.setConfidence(request.getConfidence());
        event.setTrackId(request.getTrackId());
        event.setDirection(request.getDirection());

        if (request.getTimestamp() != null &&
                !request.getTimestamp().isBlank()) {

            event.setEventTime(
                    LocalDateTime.parse(request.getTimestamp())
            );

        } else {
            event.setEventTime(LocalDateTime.now());
        }

//        return eventRepository.save(event);

        AttendanceEvent savedEvent = eventRepository.save(event);

        attendanceService.processAttendance(savedEvent);

        return savedEvent;
    }

    public List<AttendanceEvent> getAllEvents() {
        return eventRepository.findAll();
    }

    public List<AttendanceEvent> getStudentEvents(String studentId) {
        return eventRepository.findByStudentId(studentId);
    }


}