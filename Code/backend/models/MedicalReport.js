import mongoose from 'mongoose';

const medicalReportSchema = new mongoose.Schema({
  patientId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  doctorId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  radiologistId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  imageUrl: {
    type: String,
    required: false
  },
  riskScore: {
    type: Number,
    min: 0,
    max: 100
  },
  summary: {
    type: String
  },
  recommendedNextSteps: {
    type: String
  },
  status: {
    type: String,
    enum: ['pending', 'analyzed', 'reviewed'],
    default: 'pending'
  }
}, {
  timestamps: true
});

const MedicalReport = mongoose.model('MedicalReport', medicalReportSchema);

export default MedicalReport;
