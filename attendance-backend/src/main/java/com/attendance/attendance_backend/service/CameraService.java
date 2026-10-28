package com.attendance.attendance_backend.service;

import com.attendance.attendance_backend.entity.Camera;
import com.attendance.attendance_backend.repository.CameraRepository;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CameraService {

    private final CameraRepository cameraRepository;

    public CameraService(CameraRepository cameraRepository) {
        this.cameraRepository = cameraRepository;
    }

    public Camera createCamera(Camera camera) {

        if (cameraRepository.existsByCameraId(camera.getCameraId())) {
            throw new RuntimeException("Camera ID already exists");
        }

        return cameraRepository.save(camera);
    }

    public List<Camera> getAllCameras() {
        return cameraRepository.findAll();
    }

    public Camera getCameraByCameraId(String cameraId) {
        return cameraRepository.findByCameraId(cameraId)
                .orElseThrow(() -> new RuntimeException("Camera not found"));
    }
}