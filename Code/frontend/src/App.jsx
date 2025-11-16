import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';

// Auth
import Login from './components/Login';

// Patient Components
import PatientDashboard from './components/PatientDashboard';
import PatientReportView from './components/PatientReportView';
import PatientFAQ from './components/PatientFAQ';

// Radiologist Components
import RadiologistWorklist from './components/RadiologistWorklist';
import RadiologistReportInterface from './components/RadiologistReportInterface';
import RadiologistArchived from './components/RadiologistArchived';
import RadiologistUpload from './components/RadiologistUpload';

// Doctor Components
import DoctorDashboard from './components/DoctorDashboard';
import DoctorReportView from './components/DoctorReportView';

function App() {
  const [user, setUser] = useState(null);

  const handleLogin = (userData) => {
    setUser(userData);
  };

  const handleLogout = () => {
    setUser(null);
  };

  return (
    <Router>
      <div className="app">
        {user && (
          <nav className="top-nav">
            <span>Logged in as: {user.email} ({user.role})</span>
            <button onClick={handleLogout} className="btn-logout">Logout</button>
          </nav>
        )}
        
        <Routes>
          {/* Login Route */}
          <Route path="/login" element={<Login onLogin={handleLogin} />} />
          
          {/* Patient Routes */}
          <Route 
            path="/patient/dashboard" 
            element={user?.role === 'patient' ? <PatientDashboard user={user} /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/patient/report/:id" 
            element={user?.role === 'patient' ? <PatientReportView /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/patient/faq" 
            element={user?.role === 'patient' ? <PatientFAQ /> : <Navigate to="/login" />} 
          />
          
          {/* Radiologist Routes */}
          <Route 
            path="/radiologist/worklist" 
            element={user?.role === 'radiologist' ? <RadiologistWorklist user={user} /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/radiologist/upload" 
            element={user?.role === 'radiologist' ? <RadiologistUpload user={user} /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/radiologist/report/:id" 
            element={user?.role === 'radiologist' ? <RadiologistReportInterface /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/radiologist/archived" 
            element={user?.role === 'radiologist' ? <RadiologistArchived user={user} /> : <Navigate to="/login" />} 
          />
          
          {/* Doctor Routes */}
          <Route 
            path="/doctor/dashboard" 
            element={user?.role === 'doctor' ? <DoctorDashboard user={user} /> : <Navigate to="/login" />} 
          />
          <Route 
            path="/doctor/report/:id" 
            element={user?.role === 'doctor' ? <DoctorReportView /> : <Navigate to="/login" />} 
          />
          
          {/* Default Route */}
          <Route path="/" element={<Navigate to="/login" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
