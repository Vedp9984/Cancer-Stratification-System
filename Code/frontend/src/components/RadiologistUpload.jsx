import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { reportAPI, userAPI } from '../services/api';
import './RadiologistUpload.css';

function RadiologistUpload({ user }) {
  const [patients, setPatients] = useState([]);
  const [doctors, setDoctors] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState('');
  const [selectedDoctor, setSelectedDoctor] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await userAPI.getAllUsers();
      const allUsers = response.data;
      setPatients(allUsers.filter(u => u.role === 'patient'));
      setDoctors(allUsers.filter(u => u.role === 'doctor'));
      
      // Auto-select first doctor if available
      const firstDoctor = allUsers.find(u => u.role === 'doctor');
      if (firstDoctor) setSelectedDoctor(firstDoctor._id);
    } catch (error) {
      console.error('Error fetching users:', error);
      setError('Failed to load users');
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Check file type
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      
      // Check file size (max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        setError('Image size should be less than 5MB');
        return;
      }

      setImageFile(file);
      setError('');
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedPatient) {
      setError('Please select a patient');
      return;
    }
    
    if (!selectedDoctor) {
      setError('Please select a doctor');
      return;
    }
    
    if (!imageFile) {
      setError('Please select an X-ray image');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Get radiologist user from users list (temporary - in production use authenticated user ID)
      const usersResponse = await userAPI.getAllUsers();
      const radiologist = usersResponse.data.find(u => u.role === 'radiologist');
      
      if (!radiologist) {
        setError('No radiologist user found. Please create a radiologist account first.');
        setLoading(false);
        return;
      }

      // Create FormData for file upload
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('patientId', selectedPatient);
      formData.append('doctorId', selectedDoctor);
      formData.append('radiologistId', radiologist._id);

      console.log('Uploading with data:', {
        patientId: selectedPatient,
        doctorId: selectedDoctor,
        radiologistId: radiologist._id,
        imageFile: imageFile.name
      });

      // Upload the report
      const response = await reportAPI.uploadReport(formData);
      
      alert('Report uploaded successfully!');
      navigate('/radiologist/worklist');
    } catch (error) {
      console.error('Error uploading report:', error);
      setError(error.response?.data?.message || error.message || 'Failed to upload report. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    navigate('/radiologist/worklist');
  };

  return (
    <div className="radiologist-upload">
      <button className="btn-back" onClick={handleCancel}>← Back to Worklist</button>

      <div className="upload-container">
        <header className="upload-header">
          <h1>📤 Upload New X-Ray Report</h1>
          <p>Upload patient X-ray images for analysis</p>
        </header>

        {error && <div className="error-message">{error}</div>}

        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-section">
            <h3>Patient Information</h3>
            
            <div className="form-group">
              <label>Select Patient *</label>
              <select 
                value={selectedPatient} 
                onChange={(e) => setSelectedPatient(e.target.value)}
                required
              >
                <option value="">-- Select a patient --</option>
                {patients.map(patient => (
                  <option key={patient._id} value={patient._id}>
                    {patient.email}
                  </option>
                ))}
              </select>
              {patients.length === 0 && (
                <p className="hint">No patients found. Please create a patient account first.</p>
              )}
            </div>

            <div className="form-group">
              <label>Assign to Doctor *</label>
              <select 
                value={selectedDoctor} 
                onChange={(e) => setSelectedDoctor(e.target.value)}
                required
              >
                <option value="">-- Select a doctor --</option>
                {doctors.map(doctor => (
                  <option key={doctor._id} value={doctor._id}>
                    {doctor.email}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-section">
            <h3>X-Ray Image Upload</h3>
            
            <div className="form-group">
              <label>Upload X-Ray Image *</label>
              <div className="file-input-wrapper">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  id="image-upload"
                  required
                />
                <label htmlFor="image-upload" className="file-input-label">
                  {imageFile ? imageFile.name : '📁 Choose X-Ray Image'}
                </label>
              </div>
              <p className="hint">Accepted formats: JPG, PNG, DICOM. Max size: 5MB</p>
            </div>

            {imagePreview && (
              <div className="image-preview">
                <h4>Image Preview</h4>
                <img src={imagePreview} alt="X-Ray Preview" />
              </div>
            )}
          </div>

          <div className="form-actions">
            <button 
              type="button" 
              className="btn-secondary" 
              onClick={handleCancel}
              disabled={loading}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              className="btn-primary"
              disabled={loading || !imageFile || !selectedPatient || !selectedDoctor}
            >
              {loading ? 'Uploading...' : '📤 Upload Report'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default RadiologistUpload;
