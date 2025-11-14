import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './RadiologistWorklist.css';

function RadiologistWorklist({ user }) {
  const [reports, setReports] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchPendingReports();
  }, []);

  const fetchPendingReports = async () => {
    try {
      const response = await reportAPI.getAllReports();
      // Filter pending reports
      const pending = response.data.filter(r => r.status === 'pending');
      setReports(pending);
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  return (
    <div className="radiologist-worklist">
      <header className="worklist-header">
        <h1>Radiologist Worklist</h1>
        <p>Welcome, {user?.email}</p>
        <div className="stats">
          <span className="stat-badge">Pending Cases: {reports.length}</span>
        </div>
      </header>

      <div className="actions-bar">
        <button className="btn-primary" onClick={() => navigate('/radiologist/upload')}>
          + Upload New Report
        </button>
        <button className="btn-secondary" onClick={() => navigate('/radiologist/archived')}>
          📁 View Archived Reports
        </button>
      </div>

      <div className="worklist-table">
        <h2>Pending Cases</h2>
        <table>
          <thead>
            <tr>
              <th>Report ID</th>
              <th>Patient</th>
              <th>Upload Date</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {reports.map((report) => (
              <tr key={report._id}>
                <td>{report._id.slice(-6)}</td>
                <td>{report.patientId?.email || 'N/A'}</td>
                <td>{new Date(report.createdAt).toLocaleDateString()}</td>
                <td>
                  <span className="status-badge pending">{report.status}</span>
                </td>
                <td>
                  <button 
                    className="btn-action"
                    onClick={() => navigate(`/radiologist/report/${report._id}`)}
                  >
                    View & Report
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {reports.length === 0 && (
          <div className="empty-state">No pending cases at the moment.</div>
        )}
      </div>
    </div>
  );
}

export default RadiologistWorklist;
