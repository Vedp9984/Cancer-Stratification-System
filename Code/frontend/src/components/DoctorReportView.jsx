import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './PatientReportView.css'; // Using PatientReportView styles

function DoctorReportView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [viewMode, setViewMode] = useState('simplified');
  const [review, setReview] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    fetchReport();
  }, [id]);

  const fetchReport = async () => {
    try {
      const response = await reportAPI.getReportById(id);
      setReport(response.data);
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
      fetchReport();
    } catch (error) {
      console.error('Error saving review:', error);
      alert('Failed to save review');
    }
  };

  if (!report) return <div className="patient-report-view"><div>Loading...</div></div>;

  const getRiskLevel = (score) => {
    if (score < 30) return { level: 'Low', color: '#4caf50', description: 'Results show minimal concern.' };
    if (score < 70) return { level: 'Moderate', color: '#ff9800', description: 'Results require monitoring.' };
    return { level: 'High', color: '#f44336', description: 'Results require immediate attention.' };
  };

  const riskInfo = getRiskLevel(report.riskScore || 0);

  return (
    <div className="patient-report-view">
      <button className="btn-back" onClick={() => navigate(-1)}>← Back</button>
      
      <div className="view-toggle">
        <button 
          className={viewMode === 'simplified' ? 'active' : ''}
          onClick={() => setViewMode('simplified')}
        >
          Simplified View
        </button>
        <button 
          className={viewMode === 'full' ? 'active' : ''}
          onClick={() => setViewMode('full')}
        >
          Full Report
        </button>
      </div>

      {viewMode === 'simplified' ? (
        <div className="simplified-view">
          <div className="risk-score-display" style={{ borderColor: riskInfo.color }}>
            <h2>Risk Score</h2>
            <div className="score-circle" style={{ background: riskInfo.color }}>
              {report.riskScore || 'N/A'}
            </div>
            <h3 style={{ color: riskInfo.color }}>{riskInfo.level} Risk</h3>
            <p>{riskInfo.description}</p>
          </div>

          <div className="summary-section">
            <h3>📋 Summary</h3>
            <p>{report.summary || 'Analysis pending...'}</p>
          </div>

          <div className="next-steps-section">
            <h3>🩺 Recommended Next Steps</h3>
            <p>{report.recommendedNextSteps || 'Please consult with doctor.'}</p>
          </div>

          {/* Doctor's Review Section - Editable for doctors */}
          <div className="doctor-review-section">
            <h3>👨‍⚕️ Doctor's Review</h3>
            {isEditing ? (
              <>
                <textarea
                  value={review}
                  onChange={(e) => setReview(e.target.value)}
                  placeholder="Enter your clinical review and recommendations..."
                  rows="6"
                  style={{
                    width: '100%',
                    padding: '15px',
                    borderRadius: '8px',
                    border: '1px solid #ddd',
                    fontFamily: 'inherit',
                    fontSize: '14px',
                    marginBottom: '10px'
                  }}
                />
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button 
                    onClick={handleSaveReview}
                    style={{
                      padding: '10px 20px',
                      background: '#4caf50',
                      color: 'white',
                      border: 'none',
                      borderRadius: '5px',
                      cursor: 'pointer'
                    }}
                  >
                    Save Review
                  </button>
                  <button 
                    onClick={() => {
                      setIsEditing(false);
                      setReview(report.doctorReview || '');
                    }}
                    style={{
                      padding: '10px 20px',
                      background: '#6c757d',
                      color: 'white',
                      border: 'none',
                      borderRadius: '5px',
                      cursor: 'pointer'
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <>
                {report.doctorReview ? (
                  <>
                    <div style={{ 
                      background: '#e8f5e91c', 
                      padding: '15px', 
                      borderRadius: '8px', 
                      border: '2px solid #4caf50',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {report.doctorReview}
                    </div>
                    {report.reviewedAt && (
                      <p style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>
                        Reviewed on: {new Date(report.reviewedAt).toLocaleString()}
                      </p>
                    )}
                    <button 
                      onClick={() => setIsEditing(true)}
                      style={{
                        marginTop: '10px',
                        padding: '10px 20px',
                        background: '#667eea',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: 'pointer'
                      }}
                    >
                      Edit Review
                    </button>
                  </>
                ) : (
                  <button 
                    onClick={() => setIsEditing(true)}
                    style={{
                      padding: '10px 20px',
                      background: '#667eea',
                      color: 'white',
                      border: 'none',
                      borderRadius: '5px',
                      cursor: 'pointer'
                    }}
                  >
                    Add Review
                  </button>
                )}
              </>
            )}
          </div>

          <div className="report-metadata">
            <h3>Report Information</h3>
            <p><strong>Patient:</strong> {report.patientId?.email || 'N/A'}</p>
            <p><strong>Report Date:</strong> {new Date(report.createdAt).toLocaleDateString()}</p>
            <p><strong>Status:</strong> {report.status}</p>
          </div>
        </div>
      ) : (
        <div className="full-view">
          <h2>Full Medical Report</h2>
          {report.imageUrl && (
            <div className="report-image">
              <img src={`http://localhost:5000${report.imageUrl}`} alt="X-Ray Report" />
            </div>
          )}
          <div className="technical-details">
            <h3>Technical Details</h3>
            <p><strong>Report ID:</strong> {report._id}</p>
            <p><strong>Patient:</strong> {report.patientId?.email || 'N/A'}</p>
            <p><strong>Radiologist:</strong> {report.radiologistId?.email || 'N/A'}</p>
            <p><strong>Doctor:</strong> {report.doctorId?.email || 'N/A'}</p>
            <p><strong>Date:</strong> {new Date(report.createdAt).toLocaleString()}</p>
            <p><strong>Risk Score:</strong> {report.riskScore || 'Pending'}</p>
            <p><strong>Summary:</strong> {report.summary || 'Pending analysis'}</p>
            <p><strong>Next Steps:</strong> {report.recommendedNextSteps || 'Pending radiologist review'}</p>
          </div>

          {/* Doctor's Review Section - Editable for doctors */}
          <div className="doctor-review-section" style={{ marginTop: '20px' }}>
            <h3>👨‍⚕️ Doctor's Review</h3>
            {isEditing ? (
              <>
                <textarea
                  value={review}
                  onChange={(e) => setReview(e.target.value)}
                  placeholder="Enter your clinical review and recommendations..."
                  rows="6"
                  style={{
                    width: '100%',
                    padding: '15px',
                    borderRadius: '8px',
                    border: '1px solid #ddd',
                    fontFamily: 'inherit',
                    fontSize: '14px',
                    marginBottom: '10px'
                  }}
                />
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button 
                    onClick={handleSaveReview}
                    style={{
                      padding: '10px 20px',
                      background: '#4caf50',
                      color: 'white',
                      border: 'none',
                      borderRadius: '5px',
                      cursor: 'pointer'
                    }}
                  >
                    Save Review
                  </button>
                  <button 
                    onClick={() => {
                      setIsEditing(false);
                      setReview(report.doctorReview || '');
                    }}
                    style={{
                      padding: '10px 20px',
                      background: '#6c757d',
                      color: 'white',
                      border: 'none',
                      borderRadius: '5px',
                      cursor: 'pointer'
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <>
                {report.doctorReview ? (
                  <>
                    <div style={{ 
                      background: '#e8f5e91c', 
                      padding: '15px', 
                      borderRadius: '8px', 
                      border: '2px solid #4caf50',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {report.doctorReview}
                    </div>
                    {report.reviewedAt && (
                      <p style={{ fontSize: '12px', color: '#666', marginTop: '8px' }}>
                        Reviewed on: {new Date(report.reviewedAt).toLocaleString()}
                      </p>
                    )}
                    <button 
                      onClick={() => setIsEditing(true)}
                      style={{
                        marginTop: '10px',
                        padding: '10px 20px',
                        background: '#667eea',
                        color: 'white',
                        border: 'none',
                        borderRadius: '5px',
                        cursor: 'pointer'
                      }}
                    >
                      Edit Review
                    </button>
                  </>
                ) : (
                  <button 
                    onClick={() => setIsEditing(true)}
                    style={{
                      padding: '10px 20px',
                      background: '#667eea',
                      color: 'white',
                      border: 'none',
                      borderRadius: '5px',
                      cursor: 'pointer'
                    }}
                  >
                    Add Review
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default DoctorReportView;
