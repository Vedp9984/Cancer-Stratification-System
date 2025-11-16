import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './PatientDashboard.css';

function PatientDashboard({ user }) {
  const [reports, setReports] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchReports();
  }, [user]);

  const fetchReports = async () => {
    try {
      // Fetch reports filtered by user ID and role
      const response = await reportAPI.getAllReports(user._id, user.role);
      setReports(response.data);
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  const getRiskColor = (score) => {
    if (score < 30) return '#4caf50';
    if (score < 70) return '#ff9800';
    return '#f44336';
  };

  return (
    <div className="patient-dashboard">
      <header className="dashboard-header">
        <h1>Patient Dashboard</h1>
        <p>Welcome, {user?.email}</p>
      </header>

      <div className="notifications">
        <h3>📢 Recent Notifications</h3>
        <div className="notification-list">
          <div className="notification-item">New report available for review</div>
        </div>
      </div>

      <div className="reports-section">
        <h2>My Medical Reports</h2>
        <div className="reports-grid">
          {reports.map((report) => (
            <div key={report._id} className="report-card">
              <div className="report-header">
                <span className="report-date">
                  {new Date(report.createdAt).toLocaleDateString()}
                </span>
                <span 
                  className="risk-badge"
                  style={{ backgroundColor: getRiskColor(report.riskScore || 0) }}
                >
                  Risk: {report.riskScore || 'Pending'}
                </span>
              </div>
              <div className="report-status">Status: {report.status}</div>
              <button 
                className="btn-view"
                onClick={() => navigate(`/patient/report/${report._id}`)}
              >
                View Report
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="quick-actions">
        <button onClick={() => navigate('/patient/faq')} className="btn-secondary">
          📚 View FAQ
        </button>
      </div>
    </div>
  );
}

export default PatientDashboard;
