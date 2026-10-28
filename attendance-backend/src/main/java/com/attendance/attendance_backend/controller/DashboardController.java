package com.attendance.attendance_backend.controller;

import com.attendance.attendance_backend.dto.DashboardSummary;
import com.attendance.attendance_backend.service.DashboardService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/dashboard")
public class DashboardController {

    private final DashboardService dashboardService;

    public DashboardController(
            DashboardService dashboardService) {

        this.dashboardService = dashboardService;
    }

    @GetMapping("/summary")
    public ResponseEntity<DashboardSummary> getSummary() {

        return ResponseEntity.ok(
                dashboardService.getSummary()
        );
    }
}