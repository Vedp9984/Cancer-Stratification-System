import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// User API
export const userAPI = {
  getAllUsers: () => api.get('/users'),
  getUserById: (id) => api.get(`/users/${id}`),
  createUser: (userData) => api.post('/users', userData),
  login: (email, password) => api.post('/users/login', { email, password }),
};

// Medical Report API
export const reportAPI = {
  getAllReports: () => api.get('/reports'),
  getReportById: (id) => api.get(`/reports/${id}`),
  createReport: (reportData) => api.post('/reports', reportData),
  updateReport: (id, reportData) => api.patch(`/reports/${id}`, reportData),
  getReportsByPatient: (patientId) => api.get(`/reports?patientId=${patientId}`),
  getReportsByDoctor: (doctorId) => api.get(`/reports?doctorId=${doctorId}`),
  uploadReport: (formData) => api.post('/reports/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  }),
};

export default api;
