import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './DoctorDashboard.css';

function DoctorDashboard({ user }) {
  const [reports, setReports] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    try {
      const response = await reportAPI.getAllReports();
      // Sort by risk score (high to low)
      const sorted = response.data.sort((a, b) => (b.riskScore || 0) - (a.riskScore || 0));
      setReports(sorted);
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  const getUrgencyClass = (score) => {
    if (score >= 70) return 'urgent';
    if (score >= 30) return 'moderate';
    return 'low';
  };

  const getPriorityLabel = (score) => {
    if (score >= 70) return '🔴 High Priority';
    if (score >= 30) return '🟡 Medium Priority';
    return '🟢 Low Priority';
  };

  return (
    <div className="doctor-dashboard">
      <header className="dashboard-header">
        <h1>Doctor Dashboard</h1>
        <p>Welcome, Dr. {user?.email}</p>
      </header>

      <div className="stats-overview">
        <div className="stat-card urgent">
          <h3>{reports.filter(r => r.riskScore >= 70).length}</h3>
          <p>High Priority</p>
        </div>
        <div className="stat-card moderate">
          <h3>{reports.filter(r => r.riskScore >= 30 && r.riskScore < 70).length}</h3>
          <p>Medium Priority</p>
        </div>
        <div className="stat-card low">
          <h3>{reports.filter(r => r.riskScore < 30).length}</h3>
          <p>Low Priority</p>
        </div>
        <div className="stat-card total">
          <h3>{reports.length}</h3>
          <p>Total Reports</p>
        </div>
      </div>

      <div className="reports-section">
        <h2>Patient Reports (Prioritized by Risk)</h2>
        <div className="reports-list">
          {reports.map((report) => (
            <div 
              key={report._id} 
              className={`report-item ${getUrgencyClass(report.riskScore || 0)}`}
            >
              <div className="report-priority">
                {getPriorityLabel(report.riskScore || 0)}
              </div>
              <div className="report-content">
                <div className="report-info">
                  <h3>Patient: {report.patientId?.email || 'N/A'}</h3>
                  <p>Report ID: {report._id.slice(-6)}</p>
                  <p>Date: {new Date(report.createdAt).toLocaleDateString()}</p>
                  <p>Status: {report.status}</p>
                </div>
                <div className="report-score">
                  <div className="score-display">
                    {report.riskScore || 'N/A'}
                  </div>
                  <p>Risk Score</p>
                </div>
              </div>
              <button 
                className="btn-review"
                onClick={() => navigate(`/doctor/report/${report._id}`)}
              >
                Review Report
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default DoctorDashboard;
