import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './DoctorReportView.css';

function DoctorReportView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [notes, setNotes] = useState('');
  const [showFullReport, setShowFullReport] = useState(false);

  useEffect(() => {
    fetchReport();
  }, [id]);

  const fetchReport = async () => {
    try {
      const response = await reportAPI.getReportById(id);
      setReport(response.data);
    } catch (error) {
      console.error('Error fetching report:', error);
    }
  };

  const handleSaveNotes = async () => {
    try {
      await reportAPI.updateReport(id, { 
        status: 'reviewed',
        doctorNotes: notes 
      });
      alert('Notes saved successfully!');
      navigate(-1);
    } catch (error) {
      console.error('Error saving notes:', error);
    }
  };

  if (!report) return <div>Loading...</div>;

  const getRiskInfo = (score) => {
    if (score < 30) return { level: 'Low Risk', color: '#4caf50', icon: '🟢' };
    if (score < 70) return { level: 'Moderate Risk', color: '#ff9800', icon: '🟡' };
    return { level: 'High Risk', color: '#f44336', icon: '🔴' };
  };

  const riskInfo = getRiskInfo(report.riskScore || 0);

  return (
    <div className="doctor-report-view">
      <button className="btn-back" onClick={() => navigate(-1)}>← Back to Dashboard</button>

      <div className="report-layout">
        {/* Critical Information Panel */}
        <div className="critical-panel">
          <h2>Critical Information at a Glance</h2>
          
          <div className="risk-summary" style={{ borderColor: riskInfo.color }}>
            <div className="risk-icon">{riskInfo.icon}</div>
            <div className="risk-details">
              <h3 style={{ color: riskInfo.color }}>{riskInfo.level}</h3>
              <div className="risk-score-large">
                Score: {report.riskScore || 'N/A'}
              </div>
            </div>
          </div>

          <div className="patient-info">
            <h3>Patient Information</h3>
            <p><strong>Patient ID:</strong> {report.patientId?._id || 'N/A'}</p>
            <p><strong>Email:</strong> {report.patientId?.email || 'N/A'}</p>
            <p><strong>Report Date:</strong> {new Date(report.createdAt).toLocaleDateString()}</p>
            <p><strong>Status:</strong> {report.status}</p>
          </div>

          <div className="quick-summary">
            <h3>📋 Summary</h3>
            <p>{report.summary || 'No summary available yet.'}</p>
          </div>

          <div className="recommendations">
            <h3>🩺 Recommended Next Steps</h3>
            <p>{report.recommendedNextSteps || 'Pending radiologist review.'}</p>
          </div>
        </div>

        {/* Full Report Section */}
        <div className="full-report-panel">
          <div className="panel-header">
            <h2>Full Report Details</h2>
            <button 
              className="btn-toggle"
              onClick={() => setShowFullReport(!showFullReport)}
            >
              {showFullReport ? 'Hide' : 'Show'} Full Report
            </button>
          </div>

          {showFullReport && (
            <div className="full-report-content">
              {report.imageUrl && (
                <div className="xray-image">
                  <h3>X-Ray Image</h3>
                  <img src={`http://localhost:5000${report.imageUrl}`} alt="X-Ray" />
                </div>
              )}

              <div className="technical-details">
                <h3>Technical Details</h3>
                <p><strong>Report ID:</strong> {report._id}</p>
                <p><strong>Radiologist ID:</strong> {report.radiologistId}</p>
                <p><strong>Doctor ID:</strong> {report.doctorId}</p>
                <p><strong>Created:</strong> {new Date(report.createdAt).toLocaleString()}</p>
                <p><strong>Updated:</strong> {new Date(report.updatedAt).toLocaleString()}</p>
              </div>
            </div>
          )}

          {/* Doctor's Notes Section */}
          <div className="doctor-notes">
            <h3>Doctor's Notes</h3>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add your clinical notes, treatment plan, or follow-up instructions..."
              rows="8"
            />
            <div className="notes-actions">
              <button className="btn-secondary" onClick={() => navigate(-1)}>
                Cancel
              </button>
              <button className="btn-primary" onClick={handleSaveNotes}>
                Save Notes & Mark Reviewed
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DoctorReportView;
