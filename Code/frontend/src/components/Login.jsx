import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { userAPI } from '../services/api';
import './Login.css';

function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [role, setRole] = useState('patient');
  const [isSignup, setIsSignup] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isSignup) {
        // Create account
        if (password !== confirmPassword) {
          setError('Passwords do not match');
          setLoading(false);
          return;
        }
        
        const response = await userAPI.createUser({ email, password, role });
        alert('Account created successfully! Please login.');
        setIsSignup(false);
        setPassword('');
        setConfirmPassword('');
      } else {
        // Login - verify credentials against database
        const response = await userAPI.login(email, password);
        const userData = response.data;
        
        // Pass complete user data including _id
        onLogin(userData);
        
        // Navigate based on role
        if (userData.role === 'patient') navigate('/patient/dashboard');
        if (userData.role === 'radiologist') navigate('/radiologist/worklist');
        if (userData.role === 'doctor') navigate('/doctor/dashboard');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setIsSignup(!isSignup);
    setError('');
    setPassword('');
    setConfirmPassword('');
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h1>Cancer Stratification Portal</h1>
        <h2>{isSignup ? 'Create Account' : 'Login'}</h2>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              required
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
            />
          </div>
          {isSignup && (
            <div className="form-group">
              <label>Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Re-enter your password"
                required
              />
            </div>
          )}
          {isSignup && (
            <div className="form-group">
              <label>Role</label>
              <select value={role} onChange={(e) => setRole(e.target.value)} required>
                <option value="patient">Patient</option>
                <option value="radiologist">Radiologist</option>
                <option value="doctor">Doctor</option>
              </select>
            </div>
          )}
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Processing...' : (isSignup ? 'Create Account' : 'Login')}
          </button>
        </form>
        
        <div className="toggle-mode">
          <p>
            {isSignup ? 'Already have an account?' : "Don't have an account?"}
            <button type="button" onClick={toggleMode} className="btn-link">
              {isSignup ? 'Login here' : 'Create account'}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Login;
