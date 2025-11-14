import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './PatientReportView.css';

function PatientReportView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [viewMode, setViewMode] = useState('simplified'); // 'simplified' or 'full'

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

  if (!report) return <div>Loading...</div>;

  const getRiskLevel = (score) => {
    if (score < 30) return { level: 'Low', color: '#4caf50', description: 'Your results show minimal concern.' };
    if (score < 70) return { level: 'Moderate', color: '#ff9800', description: 'Your results require monitoring.' };
    return { level: 'High', color: '#f44336', description: 'Your results require immediate attention.' };
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
            <h2>Your Risk Score</h2>
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
            <p>{report.recommendedNextSteps || 'Please consult with your doctor.'}</p>
          </div>

          <div className="report-metadata">
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
            <p><strong>Date:</strong> {new Date(report.createdAt).toLocaleString()}</p>
            <p><strong>Risk Score:</strong> {report.riskScore || 'Pending'}</p>
            <p><strong>Summary:</strong> {report.summary || 'Pending analysis'}</p>
            <p><strong>Next Steps:</strong> {report.recommendedNextSteps || 'Pending radiologist review'}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default PatientReportView;
