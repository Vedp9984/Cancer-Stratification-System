# Cancer Stratification System - Frontend Documentation

## 🎯 Overview
A comprehensive MERN stack application for cancer stratification through X-Ray image analysis with role-based interfaces for Patients, Radiologists, and Doctors.

## 🚀 Running the Application

### Backend (Port 5000)
```bash
cd Code/backend
npm run dev
```

### Frontend (Port 5173)
```bash
cd Code/frontend
npm run dev
```

Visit: http://localhost:5173

## 👥 User Roles & Screens

### 🔵 Patient Screens
1. **Login Screen** (`/login`)
   - Secure authentication
   - Role selection

2. **Patient Dashboard** (`/patient/dashboard`)
   - Recent reports overview
   - Notifications panel
   - Quick access to reports

3. **Simplified Report View** (`/patient/report/:id`)
   - Risk score visualization (color-coded)
   - Easy-to-understand summary
   - Recommended next steps
   - Toggle between simplified and full view

4. **Full Report View**
   - Original X-ray image
   - Technical details
   - Complete medical information

5. **FAQ Page** (`/patient/faq`)
   - Educational content
   - Common questions answered
   - Contact information

### 🟢 Radiologist Screens
1. **Login Screen** (`/login`)
   - Secure authentication

2. **Worklist Screen** (`/radiologist/worklist`)
   - Pending cases list
   - Quick stats
   - Upload new reports option

3. **Image Viewer & Reporting Interface** (`/radiologist/report/:id`)
   - Split-screen layout
   - X-ray image viewer with tools (zoom, rotate, contrast)
   - Structured report template
   - Clinical findings entry
   - Impression and recommendations

4. **Archived Reports Screen** (`/radiologist/archived`)
   - Searchable history
   - Filter by date, patient
   - Access past reports

### 🔴 Doctor Screens
1. **Login Screen** (`/login`)
   - Secure authentication

2. **Patient Reports Dashboard** (`/doctor/dashboard`)
   - Prioritized by risk score
   - Color-coded urgency (High/Medium/Low)
   - Quick statistics overview
   - Sorted by priority

3. **Report View** (`/doctor/report/:id`)
   - Critical information at a glance
   - Risk assessment visualization
   - Patient information panel
   - Summary and recommendations
   - Full report toggle
   - Doctor's notes section
   - Mark as reviewed functionality

## 📁 Project Structure

```
Code/
├── backend/
│   ├── server.js              # Express server
│   ├── config/
│   │   └── db.js             # MongoDB connection
│   ├── models/
│   │   ├── User.js           # User schema
│   │   └── MedicalReport.js  # Report schema
│   └── routes/
│       ├── userRoutes.js     # User API endpoints
│       └── reportRoutes.js   # Report API endpoints
│
└── frontend/
    ├── src/
    │   ├── App.jsx           # Main app with routing
    │   ├── services/
    │   │   └── api.js        # API service layer
    │   └── components/
    │       ├── Login.jsx                      # Shared login
    │       ├── PatientDashboard.jsx          # Patient screens
    │       ├── PatientReportView.jsx
    │       ├── PatientFAQ.jsx
    │       ├── RadiologistWorklist.jsx       # Radiologist screens
    │       ├── RadiologistReportInterface.jsx
    │       ├── RadiologistArchived.jsx
    │       ├── DoctorDashboard.jsx           # Doctor screens
    │       └── DoctorReportView.jsx
    └── .env                   # Environment variables
```

## 🔑 Key Features

### For Patients
- ✅ Color-coded risk scores (Green/Yellow/Red)
- ✅ Simplified vs Full report toggle
- ✅ Educational FAQ section
- ✅ Notifications system
- ✅ Easy-to-understand summaries

### For Radiologists
- ✅ Worklist management
- ✅ Image viewer with tools
- ✅ Structured report templates
- ✅ Searchable archive
- ✅ Pending cases tracking

### For Doctors
- ✅ Priority-based dashboard
- ✅ Risk-stratified patient list
- ✅ Quick access to critical information
- ✅ Clinical notes capability
- ✅ Review tracking

## 🎨 Color Coding System

- 🔴 **High Risk (70-100)**: Red - Requires immediate attention
- 🟡 **Moderate Risk (30-69)**: Orange/Yellow - Requires monitoring
- 🟢 **Low Risk (0-29)**: Green - Minimal concern

## 🔌 API Endpoints

### Users
- `GET /api/users` - Get all users
- `POST /api/users` - Create new user
- `GET /api/users/:id` - Get user by ID

### Reports
- `GET /api/reports` - Get all reports
- `POST /api/reports` - Create new report
- `GET /api/reports/:id` - Get report by ID
- `PATCH /api/reports/:id` - Update report (ML results, doctor notes)

## 🗄️ Database Schema

### Users Collection
```javascript
{
  email: String,
  password: String,
  role: 'radiologist' | 'patient' | 'doctor',
  medicalReports: [ObjectId],
  timestamps: true
}
```

### Medical Reports Collection
```javascript
{
  patientId: ObjectId,
  doctorId: ObjectId,
  radiologistId: ObjectId,
  imageUrl: String,
  riskScore: Number (0-100),
  summary: String,
  recommendedNextSteps: String,
  status: 'pending' | 'analyzed' | 'reviewed',
  timestamps: true
}
```

## 🚀 Next Steps

1. **ML Integration**: Connect the ML pipeline to analyze X-rays and generate risk scores
2. **Image Upload**: Implement file upload functionality for X-ray images
3. **Authentication**: Add proper JWT-based authentication
4. **Real-time Updates**: Implement WebSocket for live notifications
5. **Report Generation**: Add PDF export functionality
6. **Advanced Search**: Implement full-text search across reports

## 📝 Testing the Application

### Quick Test Login
Use these roles in the login screen:
- **Patient**: role = "patient"
- **Radiologist**: role = "radiologist"  
- **Doctor**: role = "doctor"

(Authentication is mocked for development - implement proper auth for production)

## 🔧 Environment Variables

### Backend (.env)
```
MONGODB_URI=your_mongodb_atlas_connection_string
PORT=5000
```

### Frontend (.env)
```
VITE_API_URL=http://localhost:5000/api
```

## 📦 Dependencies

### Backend
- express
- mongoose
- dotenv
- cors
- multer

### Frontend
- react
- react-router-dom
- axios
- vite

---

**Status**: ✅ Frontend development complete with all screens for Patient, Radiologist, and Doctor roles!
