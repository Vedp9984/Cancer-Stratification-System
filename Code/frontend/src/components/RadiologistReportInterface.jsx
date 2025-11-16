import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './RadiologistReportInterface.css';

function RadiologistReportInterface() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (id) fetchReport();
  }, [id]);

  const fetchReport = async () => {
    try {
      setLoading(true);
      setError(null);
      console.log('Fetching report with ID:', id);
      const response = await reportAPI.getReportById(id);
      console.log('Report fetched:', response.data);
      setReport(response.data);
    } catch (error) {
      console.error('Error fetching report:', error);
      setError(error.response?.data?.message || 'Failed to load report');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="radiologist-interface">
        <button className="btn-back" onClick={() => navigate(-1)}>← Back to Worklist</button>
        <div style={{ padding: '40px', textAlign: 'center', background: 'white', borderRadius: '10px', marginTop: '20px' }}>
          <p>Loading report...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="radiologist-interface">
        <button className="btn-back" onClick={() => navigate(-1)}>← Back to Worklist</button>
        <div style={{ padding: '40px', textAlign: 'center', background: 'white', borderRadius: '10px', marginTop: '20px', color: '#dc3545' }}>
          <p>Error: {error}</p>
          <button onClick={fetchReport} style={{ marginTop: '20px', padding: '10px 20px', background: '#007bff', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="radiologist-interface">
        <button className="btn-back" onClick={() => navigate(-1)}>← Back to Worklist</button>
        <div style={{ padding: '40px', textAlign: 'center', background: 'white', borderRadius: '10px', marginTop: '20px' }}>
          <p>Report not found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="radiologist-interface">
      <button className="btn-back" onClick={() => navigate(-1)}>← Back to Worklist</button>

      <div className="report-view-container">
        <h2>Report Details</h2>
        
        <div className="report-info-card">
          <div className="report-metadata">
            <p><strong>Report ID:</strong> {report._id}</p>
            <p><strong>Patient:</strong> {report.patientId?.email || 'N/A'}</p>
            <p><strong>Doctor:</strong> {report.doctorId?.email || 'N/A'}</p>
            <p><strong>Date:</strong> {new Date(report.createdAt).toLocaleDateString()}</p>
            <p><strong>Status:</strong> <span className={`status-badge ${report.status}`}>{report.status}</span></p>
          </div>
        </div>

        <div className="image-viewer">
          <h3>Uploaded X-Ray Image</h3>
          <div className="image-container">
            {report.imageUrl ? (
              <img src={`http://localhost:5000${report.imageUrl}`} alt="X-Ray" />
            ) : (
              <div className="no-image">No image uploaded</div>
            )}
          </div>
        </div>

        {report.riskScore !== undefined && report.riskScore !== null && (
          <div className="ml-results-section">
            <h3>ML Analysis Results</h3>
            <div className="results-card">
              <p><strong>Risk Score:</strong> {report.riskScore}%</p>
              <p><strong>Summary:</strong> {report.summary || 'Processing...'}</p>
              <p><strong>Recommended Next Steps:</strong> {report.recommendedNextSteps || 'Pending...'}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RadiologistReportInterface;
