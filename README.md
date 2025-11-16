# Cancer Stratification System - Complete Setup Guide

**A comprehensive medical imaging analysis system using AI/ML for cancer risk assessment**

---

## Project Structure

```
NNN_for_Cancer/
├── Code/
│   ├── backend/                         # Node.js + Python Backend
│   │   ├── config/                      # Configuration files
│   │   │   └── db.js                    # MongoDB connection handler
│   │   ├── models/                      # Mongoose database schemas
│   │   │   ├── User.js                  # User model (patient/doctor/radiologist)
│   │   │   └── MedicalReport.js         # Medical report with ML metadata
│   │   ├── routes/                      # Express API endpoints
│   │   │   ├── userRoutes.js            # User auth & management routes
│   │   │   └── reportRoutes.js          # Report CRUD + ML integration
│   │   ├── utils/                       # Utility functions
│   │   │   └── mlPipeline.js            # ML pipeline integration layer
│   │   ├── uploads/                     # Uploaded X-ray images storage
│   │   │   ├── .gitkeep                 # Keep directory in git
│   │   │   └── xray-*.png               # Uploaded image files
│   │   ├── Dockerfile                   # Backend container definition
│   │   ├── .dockerignore                # Docker build exclusions
│   │   ├── .env                         # Environment variables
│   │   ├── .gitignore                   # Git exclusions
│   │   ├── server.js                    # Express server entry point
│   │   ├── package.json                 # Node.js dependencies
│   │   ├── package-lock.json            # Locked dependency versions
│   │   ├── requirements.txt             # Python dependencies
│   │   ├── risk_model.py                # Main ML pipeline (900+ lines)
│   │   ├── train_xgboost_model.py       # XGBoost model training script
│   │   ├── xgboost_risk_model.pkl       # Trained XGBoost model
│   │   ├── feature_scaler.pkl           # Feature normalization scaler
│   │   ├── mock_model.ipynb             # Jupyter notebook for testing
│   │   ├── sample_report_high_risk.png  # Sample test image (high risk)
│   │   ├── sample_report_medium_risk.png # Sample test image (medium risk)
│   │   └── x-ray-test.png               # Sample test image
│   │
│   └── frontend/                        # React + Vite Frontend
│       ├── public/                      # Static assets
│       │   └── vite.svg                 # Vite logo
│       ├── src/                         # React source code
│       │   ├── assets/                  # Images, icons, fonts
│       │   ├── components/              # React components (26 files)
│       │   │   ├── Login.jsx            # Authentication screen
│       │   │   ├── Login.css            # Login styles
│       │   │   ├── Home.jsx             # Home page component
│       │   │   ├── Home.css             # Home page styles
│       │   │   ├── PatientDashboard.jsx # Patient dashboard view
│       │   │   ├── PatientDashboard.css # Patient dashboard styles
│       │   │   ├── PatientReportView.jsx # Patient report details
│       │   │   ├── PatientReportView.css # Patient report styles
│       │   │   ├── PatientFAQ.jsx       # Patient FAQ page
│       │   │   ├── PatientFAQ.css       # FAQ styles
│       │   │   ├── DoctorDashboard.jsx  # Doctor dashboard view
│       │   │   ├── DoctorDashboard.css  # Doctor dashboard styles
│       │   │   ├── DoctorReportView.jsx # Doctor report review page
│       │   │   ├── DoctorReportView.css # Doctor report styles
│       │   │   ├── RadiologistWorklist.jsx # Radiologist work queue
│       │   │   ├── RadiologistWorklist.css # Worklist styles
│       │   │   ├── RadiologistUpload.jsx # Image upload interface
│       │   │   ├── RadiologistUpload.css # Upload styles
│       │   │   ├── RadiologistReportInterface.jsx # Report view
│       │   │   ├── RadiologistReportInterface.css # Report view styles
│       │   │   ├── RadiologistArchived.jsx # Archived reports
│       │   │   ├── RadiologistArchived.css # Archive styles
│       │   │   ├── Reports.jsx          # General reports component
│       │   │   ├── Reports.css          # Reports styles
│       │   │   ├── Users.jsx            # User management component
│       │   │   └── Users.css            # User management styles
│       │   ├── services/                # API communication layer
│       │   │   └── api.js               # Axios API client
│       │   ├── App.jsx                  # Main app component with routing
│       │   ├── App.css                  # Global app styles
│       │   ├── index.css                # Root CSS styles
│       │   └── main.jsx                 # React entry point
│       ├── Dockerfile                   # Frontend container (multi-stage)
│       ├── .dockerignore                # Docker build exclusions
│       ├── .env                         # Frontend environment variables
│       ├── .gitignore                   # Git exclusions
│       ├── nginx.conf                   # Nginx web server configuration
│       ├── package.json                 # Frontend dependencies
│       ├── package-lock.json            # Locked dependency versions
│       ├── vite.config.js               # Vite build configuration
│       ├── eslint.config.js             # ESLint configuration
│       └── index.html                   # HTML template
│
├── Docs/                                # Project documentation & diagrams
│   ├── sequence-diagrams/               # System workflow diagrams
│   │   ├── AI_Model_Prediction_Pipeline.png
│   │   ├── Complete_User_Jounney_from_upload_to_Treatment.png
│   │   ├── Doctor_Review_AI-Generated_Report.png
│   │   ├── patient_login_and_view_reports.png
│   │   ├── Radiologist_upload_X-ray_Report.png
│   │   └── Tech_Team_Monitor_AI_Model_Performance.png
│   ├── schema.sql                       # Initial database schema
│   ├── Architecture_design.png          # System architecture diagram
│   ├── Entity-relation.png              # ER diagram
│   ├── API_signture.png                 # API signature diagram
│   ├── UML.png                          # UML diagrams
│   ├── api.pdf                          # API documentation
│   └── final_report.pdf                 # Complete project report
│
├── docker-compose.yml                   # Docker orchestration file
├── .env                                 # Root environment variables
├── deploy.sh                            # Deployment automation script
├── stop.sh                              # Stop services script
├── backup.sh                            # Database backup script
├── check-requirements.sh                # System requirements checker
├── PROJECT_SETUP.md                     # This file - complete setup guide
├── START_HERE.md                        # Quick start guide
├── DEPLOYMENT_GUIDE.md                  # Comprehensive deployment guide
├── DEPLOYMENT_SUMMARY.md                # Deployment overview
└── DOCKER_README.md                     # Docker commands reference
```

---

## System Overview

### Technology Stack

**Frontend:**
- React 18 with Hooks
- Vite for build tooling
- React Router DOM for navigation
- Axios for API communication
- CSS3 for styling

**Backend:**
- Node.js with Express.js
- MongoDB with Mongoose ODM
- Multer for file uploads
- Child Process for Python integration
- CSV parsing for ML results

**Machine Learning:**
- Python 3.11+
- BioBERT (dmis-lab/biobert-v1.1) - Medical NLP
- XGBoost - Risk classification
- PyTesseract - OCR text extraction
- Transformers (Hugging Face)
- PyTorch for deep learning
- CheXpert labeling system

**Infrastructure:**
- Docker & Docker Compose
- Nginx for frontend serving
- MongoDB 7.0
- Linux (Fedora) deployment

### Key Features

- **Role-Based Access Control** (Patient, Doctor, Radiologist)
- **Automated ML Risk Assessment** (30-90 second processing)
- **Multi-Model Ensemble** (BioBERT 40%, CheXpert 30%, XGBoost 20%, Clinical 10%)
- **Real-time Report Analysis**
- **Doctor Review System**
- **Secure Authentication**
- **Image Upload & Storage**
- **Comprehensive Reporting**

---

## Quick Setup (3 Steps)

### Prerequisites
- **OS**: Fedora Linux (or any Linux with Docker support)
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 10GB free space
- **Ports**: 80, 5000, 27017 available

### Step 1: Install Docker

```bash
# Install Docker on Fedora
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER

# Apply group changes (or log out and back in)
newgrp docker

# Verify installation
docker --version
docker compose version
```

### Step 2: Deploy Application

```bash
cd ~/NNN_for_Cancer

# Check system requirements
./check-requirements.sh

# Deploy with one command
./deploy.sh
```

### Step 3: Access Application

Open browser:
- **Frontend**: http://localhost
- **Backend API**: http://localhost:5000/api

**Test Credentials:**
- Patient: `patient@test.com` / `password123`
- Doctor: `doctor@test.com` / `password123`
- Radiologist: `radiologist@test.com` / `password123`

---

## Detailed Setup Instructions

### Option A: Docker Deployment (Recommended)

**1. Clone/Navigate to Project:**
```bash
cd ~/NNN_for_Cancer
```

**2. Configure Environment:**
```bash
# Create environment file
cp .env.example .env

# Edit if needed
nano .env
```

**3. Build & Deploy:**
```bash
# Build all Docker images
docker compose build

# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

**4. Verify Deployment:**
```bash
# Test backend
curl http://localhost:5000/api/users

# Test frontend
curl http://localhost

# Access MongoDB
docker exec -it cancer-stratification-db mongosh -u admin -p adminpassword123
```

### Option B: Manual Setup (Development)

**Backend Setup:**
```bash
cd Code/backend

# Install Node.js dependencies
npm install

# Install Python dependencies
pip3 install -r requirements.txt

# Install Tesseract OCR
sudo dnf install tesseract tesseract-langpack-eng

# Create uploads directory
mkdir -p uploads

# Set environment variables
export MONGODB_URI="mongodb://localhost:27017/cancer_stratification"
export PORT=5000

# Start backend
npm start
```

**Frontend Setup:**
```bash
cd Code/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Database Setup:**
```bash
# Install MongoDB
sudo dnf install mongodb-org

# Start MongoDB
sudo systemctl start mongod

# Create database and users (use mongosh)
```

---

## Configuration

### Environment Variables (.env)

```bash
# MongoDB Configuration
MONGO_ROOT_PASSWORD=your-secure-password
MONGODB_URI=mongodb://admin:password@mongodb:27017/cancer_stratification?authSource=admin

# Backend Configuration
NODE_ENV=production
PORT=5000

# Frontend Configuration
VITE_API_URL=http://localhost:5000/api
```

### Using MongoDB Atlas (Production)

```bash
# Update .env with Atlas connection string
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/cancer_stratification?retryWrites=true&w=majority

# Comment out MongoDB service in docker-compose.yml
# Rebuild and deploy
docker compose up -d --build
```

---

## Usage Guide

### For Radiologists:
1. Login at http://localhost
2. Navigate to "Upload New Report"
3. Select patient and doctor
4. Upload X-ray image (JPEG/PNG, max 5MB)
5. Wait for ML analysis (30-90 seconds)
6. View results in worklist

### For Doctors:
1. Login and view assigned reports
2. Reports sorted by risk score (high to low)
3. Click "View Report" to see details
4. Add clinical review and recommendations
5. Save review (marks report as "reviewed")

### For Patients:
1. Login to view personal reports
2. Toggle between Simplified and Full view
3. See risk score with color coding:
   - Low Risk (<30%)
   - Moderate Risk (30-70%)
   - High Risk (>70%)
4. Read AI analysis and doctor's review
5. Access FAQ for health information

---

## ML Pipeline Architecture

### Processing Flow:
```
X-Ray Image Upload
    ↓
1. OCR Text Extraction (Tesseract)
    ↓
2. CheXpert Labeling (14 conditions)
    ↓
3. BioBERT Analysis (Medical NLP)
    ↓
4. Clinical Feature Extraction
    ↓
5. Unified Feature Vector (14 features)
    ↓
6. XGBoost Classification
    ↓
7. Ensemble Scoring
    ↓
Risk Score (0-100%) + Medical Summary
```

### ML Components:

**BioBERT (40% weight):**
- Model: dmis-lab/biobert-v1.1
- 768-dimensional embeddings
- Medical text understanding

**CheXpert (30% weight):**
- 14 pathology labels
- Keyword-based detection
- Clinical finding identification

**XGBoost (20% weight):**
- Pre-trained classification model
- 14-feature input vector
- Low/Medium/High risk prediction

**Clinical Features (10% weight):**
- Age risk indicators
- Severity markers
- Bilateral/acute conditions

---

## Common Operations

### View Logs:
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs backend -f
docker compose logs frontend -f
docker compose logs mongodb -f

# Last 100 lines
docker compose logs --tail=100
```

### Restart Services:
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart backend

# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes data)
docker compose down -v
```

### Update Application:
```bash
# After code changes
docker compose build
docker compose up -d

# Force rebuild (no cache)
docker compose build --no-cache
docker compose up -d
```

### Database Operations:
```bash
# Backup database
./backup.sh

# Access MongoDB shell
docker exec -it cancer-stratification-db mongosh -u admin -p adminpassword123

# Restore from backup
docker exec -i cancer-stratification-db mongorestore \
  --username admin \
  --password adminpassword123 \
  --authenticationDatabase admin \
  --gzip \
  --archive < /path/to/backup.archive.gz
```

### Monitor Resources:
```bash
# Container status
docker compose ps

# Resource usage
docker stats

# Health checks
curl http://localhost:5000/api/users
curl http://localhost
```

---

## Troubleshooting

### Issue: Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: Port Already in Use
```bash
# Check what's using the port
sudo lsof -i :80
sudo lsof -i :5000
sudo lsof -i :27017

# Kill process or change port in docker-compose.yml
```

### Issue: ML Pipeline Fails
```bash
# Check Python dependencies
docker exec -it cancer-stratification-backend pip list

# Check Tesseract
docker exec -it cancer-stratification-backend tesseract --version

# View detailed logs
docker compose logs backend | grep -i error
```

### Issue: Frontend 404 Errors
```bash
# Rebuild frontend
docker compose build frontend
docker compose up -d frontend

# Clear browser cache
```

### Issue: MongoDB Connection Failed
```bash
# Check MongoDB is running
docker compose ps mongodb

# Test connection
docker exec -it cancer-stratification-backend node -e "
const mongoose = require('mongoose');
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log('Connected'))
  .catch(err => console.error(err));
"
```

---

## Security Recommendations

### Production Checklist:

- [ ] **Change all default passwords** in `.env`
- [ ] **Update test user credentials** in database
- [ ] **Enable firewall:**
  ```bash
  sudo firewall-cmd --permanent --add-port=80/tcp
  sudo firewall-cmd --permanent --add-port=5000/tcp
  sudo firewall-cmd --reload
  ```
- [ ] **Set up HTTPS** with Let's Encrypt:
  ```bash
  sudo dnf install certbot python3-certbot-nginx
  sudo certbot --nginx -d yourdomain.com
  ```
- [ ] **Use MongoDB Atlas** for production database
- [ ] **Implement JWT authentication** (replace simple auth)
- [ ] **Add rate limiting** on API endpoints
- [ ] **Enable CORS** with specific origins only
- [ ] **Set up monitoring** (Prometheus, Grafana)
- [ ] **Configure log rotation**
- [ ] **Schedule automated backups:**
  ```bash
  crontab -e
  # Add: 0 2 * * * /home/user/NNN_for_Cancer/backup.sh
  ```

---

## Performance Optimization

### For Better Performance:

**1. Increase Container Resources:**
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

**2. Enable GPU for ML:**
```bash
# Install nvidia-docker
sudo dnf install nvidia-docker2

# Update docker-compose.yml backend service:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**3. Cache BioBERT Model:**
- Model loads once and stays in memory
- First request: ~30 seconds
- Subsequent requests: ~5-10 seconds

**4. Database Optimization:**
- Add indexes on frequently queried fields
- Use MongoDB Atlas with auto-scaling
- Enable connection pooling

---

## Monitoring & Maintenance

### Health Checks:
```bash
# Service health
docker compose ps

# Backend API health
curl http://localhost:5000/api/users

# Frontend health
curl http://localhost

# Database health
docker exec cancer-stratification-db mongosh --eval "db.adminCommand('ping')"
```

### Regular Maintenance:
```bash
# Weekly: Check logs for errors
docker compose logs --since 7d | grep -i error

# Monthly: Clean up unused Docker resources
docker system prune -a

# Daily: Automated backup (via cron)
./backup.sh
```

---

## Additional Resources

### Documentation Files:
- **START_HERE.md** - Quick start guide
- **DEPLOYMENT_GUIDE.md** - Comprehensive deployment instructions
- **DEPLOYMENT_SUMMARY.md** - Deployment overview
- **DOCKER_README.md** - Docker commands reference
- **Code/backend/README_risk_model.md** - ML model documentation
- **Code/frontend/README.md** - Frontend development guide

### External Links:
- [Docker Documentation](https://docs.docker.com)
- [React Documentation](https://react.dev)
- [MongoDB Documentation](https://docs.mongodb.com)
- [BioBERT Paper](https://arxiv.org/abs/1901.08746)
- [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)

---

## Getting Help

### Debug Steps:
1. Check system requirements: `./check-requirements.sh`
2. View logs: `docker compose logs -f`
3. Verify services: `docker compose ps`
4. Test connectivity: `curl http://localhost:5000/api/users`
5. Review documentation in project root

### Common Commands:
```bash
# Complete restart
docker compose down && docker compose up -d

# View specific service logs
docker compose logs [service-name] --tail=50 -f

# Execute command in container
docker exec -it cancer-stratification-backend bash

# Check container resources
docker stats cancer-stratification-backend
```

---

## License & Credits

**Project**: Cancer Stratification System using AI/ML
**Version**: 1.0.0
**Last Updated**: November 16, 2025

**Technologies Used:**
- React, Node.js, Python, MongoDB
- BioBERT, XGBoost, PyTesseract
- Docker, Nginx

**Setup Created**: November 16, 2025

---

## Quick Command Reference

```bash
# Deploy everything
./deploy.sh

# Stop everything
./stop.sh

# Check system
./check-requirements.sh

# Backup database
./backup.sh

# View logs
docker compose logs -f

# Restart services
docker compose restart

# Update after changes
docker compose build && docker compose up -d

# Access application
firefox http://localhost
```

---

**Ready to Deploy! Follow the Quick Setup section above to get started.**

For detailed instructions, see **START_HERE.md** or **DEPLOYMENT_GUIDE.md**
