package com.attendance.attendance_backend.repository;

import com.attendance.attendance_backend.entity.Camera;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface CameraRepository extends JpaRepository<Camera, Long> {

    Optional<Camera> findByCameraId(String cameraId);

    boolean existsByCameraId(String cameraId);
    long countByStatus(String status);
}