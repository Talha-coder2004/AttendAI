package com.attendance.attendance_backend.controller;

import com.attendance.attendance_backend.entity.Classroom;
import com.attendance.attendance_backend.service.ClassroomService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/classrooms")
public class ClassroomController {

    private final ClassroomService classroomService;

    public ClassroomController(ClassroomService classroomService) {
        this.classroomService = classroomService;
    }

    @PostMapping
    public ResponseEntity<Classroom> createClassroom(
            @RequestBody Classroom classroom) {

        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(classroomService.createClassroom(classroom));
    }

    @GetMapping
    public ResponseEntity<List<Classroom>> getAllClassrooms() {
        return ResponseEntity.ok(classroomService.getAllClassrooms());
    }

    @GetMapping("/{id}")
    public ResponseEntity<Classroom> getClassroomById(
            @PathVariable Long id) {

        return ResponseEntity.ok(
                classroomService.getClassroomById(id)
        );
    }
}