import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import MedicalReport from '../models/MedicalReport.js';

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

// Get all reports
router.get('/', async (req, res) => {
  try {
    const reports = await MedicalReport.find()
      .populate('patientId', 'email role')
      .populate('doctorId', 'email role')
      .populate('radiologistId', 'email role');
    res.json(reports);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
});

// Upload report with image
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
    console.log('Image saved to:', imageUrl);

    const report = new MedicalReport({
      patientId,
      doctorId,
      radiologistId,
      imageUrl,
      status: 'pending'
    });

    const newReport = await report.save();
    console.log('Report saved successfully:', newReport._id);
    res.status(201).json(newReport);
  } catch (error) {
    console.error('Error uploading report:', error.message);
    console.error('Full error:', error);
    res.status(400).json({ message: error.message, details: error.toString() });
  }
});

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
