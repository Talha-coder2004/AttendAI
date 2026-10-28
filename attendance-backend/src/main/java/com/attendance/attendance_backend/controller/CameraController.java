package com.attendance.attendance_backend.controller;

import com.attendance.attendance_backend.entity.Camera;
import com.attendance.attendance_backend.service.CameraService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/cameras")
public class CameraController {

    private final CameraService cameraService;

    public CameraController(CameraService cameraService) {
        this.cameraService = cameraService;
    }

    @PostMapping
    public ResponseEntity<Camera> createCamera(
            @RequestBody Camera camera) {

        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(cameraService.createCamera(camera));
    }

    @GetMapping
    public ResponseEntity<List<Camera>> getAllCameras() {
        return ResponseEntity.ok(cameraService.getAllCameras());
    }

    @GetMapping("/{cameraId}")
    public ResponseEntity<Camera> getCamera(
            @PathVariable String cameraId) {

        return ResponseEntity.ok(
                cameraService.getCameraByCameraId(cameraId)
        );
    }
}