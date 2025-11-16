import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import MedicalReport from '../models/MedicalReport.js';
import { runMLPipeline, generateRecommendations } from '../utils/mlPipeline.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, path.join(__dirname, '../uploads'));
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'xray-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const fileFilter = (req, file, cb) => {
  // Accept images only
  if (file.mimetype.startsWith('image/')) {
    cb(null, true);
  } else {
    cb(new Error('Only image files are allowed!'), false);
  }
};

const upload = multer({ 
  storage: storage,
  fileFilter: fileFilter,
  limits: { fileSize: 5 * 1024 * 1024 } // 5MB limit
});

// Get all reports (with optional filtering by user role)
router.get('/', async (req, res) => {
  try {
    const { userId, userRole } = req.query;
    
    let query = {};
    
    // Filter reports based on user role
    if (userId && userRole === 'patient') {
      // Patients can only see their own reports
      query.patientId = userId;
    } else if (userId && userRole === 'radiologist') {
      // Radiologists can see reports they created
      query.radiologistId = userId;
    } else if (userId && userRole === 'doctor') {
      // Doctors can only see reports assigned to them
      query.doctorId = userId;
    }
    
    const reports = await MedicalReport.find(query)
      .populate('patientId', 'email role')
      .populate('doctorId', 'email role')
      .populate('radiologistId', 'email role')
      .sort({ createdAt: -1 });
    
    res.json(reports);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Upload report with image and run ML pipeline
router.post('/upload', upload.single('image'), async (req, res) => {
  try {
    console.log('Upload request received');
    console.log('File:', req.file);
    console.log('Body:', req.body);

    if (!req.file) {
      console.error('No file uploaded');
      return res.status(400).json({ message: 'No image file uploaded' });
    }

    const { patientId, doctorId, radiologistId } = req.body;
    
    if (!patientId || !doctorId || !radiologistId) {
      console.error('Missing required fields:', { patientId, doctorId, radiologistId });
      return res.status(400).json({ message: 'Missing required fields: patientId, doctorId, or radiologistId' });
    }

    // Create image URL
    const imageUrl = `/uploads/${req.file.filename}`;
    const imagePath = req.file.path; // Full path to the uploaded file
    console.log('Image saved to:', imageUrl);
    console.log('Full image path:', imagePath);

    // Create initial report with pending status
    const report = new MedicalReport({
      patientId,
      doctorId,
      radiologistId,
      imageUrl,
      status: 'pending'
    });

    const newReport = await report.save();
    console.log('Report saved successfully:', newReport._id);

    // Run ML pipeline asynchronously (don't wait for completion)
    // This prevents timeout issues with long-running ML processing
    processReportWithML(newReport._id, imagePath).catch(error => {
      console.error('ML pipeline error for report', newReport._id, ':', error);
    });

    // Return the report immediately with pending status
    res.status(201).json(newReport);
    
  } catch (error) {
    console.error('Error uploading report:', error.message);
    console.error('Full error:', error);
    res.status(400).json({ message: error.message, details: error.toString() });
  }
});

// Async function to process report with ML pipeline
async function processReportWithML(reportId, imagePath) {
  try {
    console.log(`\nStarting ML processing for report: ${reportId}`);
    
    // Run the ML pipeline
    const mlResults = await runMLPipeline(imagePath);
    
    // Generate recommendations based on risk and findings
    const recommendations = generateRecommendations(
      mlResults.riskCategory,
      mlResults.positiveFindings
    );
    
    // Update the report with ML results
    await MedicalReport.findByIdAndUpdate(reportId, {
      riskScore: mlResults.riskScore,
      summary: mlResults.summary,
      recommendedNextSteps: recommendations,
      status: 'analyzed',
      mlMetadata: {
        chexpertScore: mlResults.chexpertScore,
        biobertScore: mlResults.biobertScore,
        xgboostScore: mlResults.xgboostScore,
        clinicalScore: mlResults.clinicalScore,
        positiveFindings: mlResults.positiveFindings,
        processedAt: new Date()
      }
    });
    
    console.log(`✓ Report ${reportId} updated with ML results`);
    console.log(`  Risk Score: ${mlResults.riskScore}%`);
    console.log(`  Status: analyzed`);
    
  } catch (error) {
    console.error(`✗ ML processing failed for report ${reportId}:`, error);
    
    // Update report with error status
    await MedicalReport.findByIdAndUpdate(reportId, {
      status: 'pending',
      summary: 'ML analysis failed. Manual review required.',
      mlMetadata: {
        error: error.message,
        failedAt: new Date()
      }
    });
  }
}

// Create report (without image)
router.post('/', async (req, res) => {
  try {
    const report = new MedicalReport(req.body);
    const newReport = await report.save();
    res.status(201).json(newReport);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

// Get report by ID
router.get('/:id', async (req, res) => {
  try {
    const report = await MedicalReport.findById(req.params.id)
      .populate('patientId', 'email role')
      .populate('doctorId', 'email role')
      .populate('radiologistId', 'email role');
    if (!report) return res.status(404).json({ message: 'Report not found' });
    res.json(report);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Update report (for ML results)
router.patch('/:id', async (req, res) => {
  try {
    const report = await MedicalReport.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true }
    );
    if (!report) return res.status(404).json({ message: 'Report not found' });
    res.json(report);
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
});

export default router;
