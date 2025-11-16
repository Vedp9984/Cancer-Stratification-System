import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { reportAPI } from '../services/api';
import './RadiologistArchived.css';

function RadiologistArchived({ user }) {
  const [reports, setReports] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    fetchArchivedReports();
  }, [user]);

  const fetchArchivedReports = async () => {
    try {
      // Fetch reports created by this radiologist
      const response = await reportAPI.getAllReports(user._id, user.role);
      // Filter analyzed/reviewed reports
      const archived = response.data.filter(r => r.status !== 'pending');
      setReports(archived);
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  const filteredReports = reports.filter(report =>
    report._id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    report.patientId?.email?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="radiologist-archived">
      <button className="btn-back" onClick={() => navigate(-1)}>← Back</button>

      <header className="archived-header">
        <h1>Archived Reports</h1>
        <p>Search and review past reports</p>
      </header>

      <div className="search-bar">
        <input
          type="text"
          placeholder="🔍 Search by Report ID or Patient Email..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      <div className="archived-table">
        <table>
          <thead>
            <tr>
              <th>Report ID</th>
              <th>Patient</th>
              <th>Date</th>
              <th>Status</th>
              <th>Risk Score</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredReports.map((report) => (
              <tr key={report._id}>
                <td>{report._id.slice(-6)}</td>
                <td>{report.patientId?.email || 'N/A'}</td>
                <td>{new Date(report.createdAt).toLocaleDateString()}</td>
                <td>
                  <span className={`status-badge ${report.status}`}>
                    {report.status}
                  </span>
                </td>
                <td>{report.riskScore || 'N/A'}</td>
                <td>
                  <button 
                    className="btn-view"
                    onClick={() => navigate(`/radiologist/report/${report._id}`)}
                  >
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filteredReports.length === 0 && (
          <div className="empty-state">No archived reports found.</div>
        )}
      </div>
    </div>
  );
}

export default RadiologistArchived;
