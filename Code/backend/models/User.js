import mongoose from 'mongoose';

const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true
  },
  password: {
    type: String,
    required: true
  },
  role: {
    type: String,
    enum: ['radiologist', 'patient', 'doctor'],
    required: true
  },
  medicalReports: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'MedicalReport'
  }]
}, {
  timestamps: true
});

const User = mongoose.model('User', userSchema);

export default User;
