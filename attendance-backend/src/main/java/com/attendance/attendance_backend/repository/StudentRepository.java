package com.attendance.attendance_backend.repository;

import com.attendance.attendance_backend.entity.Student;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;


public interface StudentRepository extends JpaRepository<Student, Long> {

    Optional<Student> findByStudentId(String studentId);

    Optional<Student> findByRegisterNumber(String registerNumber);
    long countByStatus(String status);
}