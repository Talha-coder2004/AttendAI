package com.attendance.attendance_backend.repository;

import com.attendance.attendance_backend.entity.Classroom;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ClassroomRepository extends JpaRepository<Classroom, Long> {

    Optional<Classroom> findByRoomNumber(String roomNumber);
}