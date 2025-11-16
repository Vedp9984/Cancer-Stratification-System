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
  const [findings, setFindings] = useState('');
  const [impression, setImpression] = useState('');
  const [recommendations, setRecommendations] = useState('');

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await reportAPI.updateReport(id, {
        summary: `Findings: ${findings}\nImpression: ${impression}`,
        recommendedNextSteps: recommendations,
        status: 'analyzed'
      });
      alert('Report submitted successfully!');
      navigate('/radiologist/worklist');
    } catch (error) {
      console.error('Error submitting report:', error);
      alert('Error submitting report');
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

      <div className="interface-layout">
        <div className="image-viewer">
          <h2>X-Ray Image Viewer</h2>
          <div className="image-container">
            {report.imageUrl ? (
              <img src={`http://localhost:5000${report.imageUrl}`} alt="X-Ray" />
            ) : (
              <div className="no-image">No image uploaded</div>
            )}
          </div>
          <div className="viewer-tools">
            <button>🔍 Zoom In</button>
            <button>🔍 Zoom Out</button>
            <button>↻ Rotate</button>
            <button>◐ Adjust Contrast</button>
          </div>
        </div>

        <div className="reporting-form">
          <h2>Structured Report Template</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-section">
              <label>Clinical Findings</label>
              <textarea
                value={findings}
                onChange={(e) => setFindings(e.target.value)}
                placeholder="Describe the radiological findings..."
                rows="6"
                required
              />
            </div>

            <div className="form-section">
              <label>Impression</label>
              <textarea
                value={impression}
                onChange={(e) => setImpression(e.target.value)}
                placeholder="Provide your clinical impression..."
                rows="4"
                required
              />
            </div>

            <div className="form-section">
              <label>Recommendations</label>
              <textarea
                value={recommendations}
                onChange={(e) => setRecommendations(e.target.value)}
                placeholder="Suggest next steps or follow-up procedures..."
                rows="4"
                required
              />
            </div>

            <div className="report-metadata">
              <p><strong>Report ID:</strong> {report._id}</p>
              <p><strong>Patient:</strong> {report.patientId?.email || 'N/A'}</p>
              <p><strong>Doctor:</strong> {report.doctorId?.email || 'N/A'}</p>
              <p><strong>Date:</strong> {new Date(report.createdAt).toLocaleDateString()}</p>
            </div>

            <div className="form-actions">
              <button type="button" className="btn-secondary" onClick={() => navigate(-1)}>
                Save Draft
              </button>
              <button type="submit" className="btn-primary">
                Submit Report
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default RadiologistReportInterface;
