import axios from "axios";

const api = axios.create({
    baseURL: "http://localhost:8080/api",
    headers: {
        "Content-Type": "application/json"
    }
});

export const getDashboardSummary = () =>
    api.get("/dashboard/summary");

export const getTodayAttendance = () =>
    api.get("/attendance/today");

export const getStudents = () =>
    api.get("/students");

export const getClassrooms = () =>
    api.get("/classrooms");

export const getCameras = () =>
    api.get("/cameras");

export default api;