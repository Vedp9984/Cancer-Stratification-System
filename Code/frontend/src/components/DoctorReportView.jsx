import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './DoctorReportView.css';

function DoctorReportView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [review, setReview] = useState('');
  const [showFullReport, setShowFullReport] = useState(false);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    fetchReport();
  }, [id]);

  const fetchReport = async () => {
    try {
      const response = await reportAPI.getReportById(id);
      setReport(response.data);
      // Load existing review if present
      if (response.data.doctorReview) {
        setReview(response.data.doctorReview);
      }
    } catch (error) {
      console.error('Error fetching report:', error);
    }
  };

  const handleSaveReview = async () => {
    try {
      await reportAPI.updateReport(id, { 
        status: 'reviewed',
        doctorReview: review,
        reviewedAt: new Date().toISOString()
      });
      alert('Review saved successfully!');
      setIsEditing(false);
      fetchReport(); // Refresh to show updated data
    } catch (error) {
      console.error('Error saving review:', error);
      alert('Failed to save review');
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
                <p><strong>Radiologist:</strong> {report.radiologistId?.email || 'N/A'}</p>
                <p><strong>Doctor:</strong> {report.doctorId?.email || 'N/A'}</p>
                <p><strong>Created:</strong> {new Date(report.createdAt).toLocaleString()}</p>
                <p><strong>Updated:</strong> {new Date(report.updatedAt).toLocaleString()}</p>
              </div>
            </div>
          )}

          {/* Doctor's Review Section */}
          <div className="doctor-notes">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3>Doctor's Review</h3>
              {report.doctorReview && !isEditing && (
                <button 
                  className="btn-secondary" 
                  onClick={() => setIsEditing(true)}
                  style={{ fontSize: '14px', padding: '8px 16px' }}
                >
                  ✏️ Edit Review
                </button>
              )}
            </div>
            
            {report.reviewedAt && (
              <p style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
                Last reviewed: {new Date(report.reviewedAt).toLocaleString()}
              </p>
            )}

            {(!report.doctorReview || isEditing) ? (
              <>
                <textarea
                  value={review}
                  onChange={(e) => setReview(e.target.value)}
                  placeholder="Add your clinical review, diagnosis, treatment plan, or follow-up instructions..."
                  rows="8"
                />
                <div className="notes-actions">
                  <button className="btn-secondary" onClick={() => {
                    if (report.doctorReview) {
                      setReview(report.doctorReview);
                      setIsEditing(false);
                    } else {
                      navigate(-1);
                    }
                  }}>
                    Cancel
                  </button>
                  <button 
                    className="btn-primary" 
                    onClick={handleSaveReview}
                    disabled={!review.trim()}
                  >
                    {report.doctorReview ? 'Update Review' : 'Save Review & Mark Reviewed'}
                  </button>
                </div>
              </>
            ) : (
              <div style={{ 
                background: '#f8f9fa', 
                padding: '15px', 
                borderRadius: '5px', 
                border: '1px solid #ddd',
                whiteSpace: 'pre-wrap'
              }}>
                {report.doctorReview}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DoctorReportView;
