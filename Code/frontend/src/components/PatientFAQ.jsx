import { useNavigate } from 'react-router-dom';
import './PatientFAQ.css';

function PatientFAQ() {
  const navigate = useNavigate();

  const faqs = [
    {
      question: "What does my risk score mean?",
      answer: "The risk score is a number between 0-100 that indicates the likelihood of cancer. Lower scores (0-30) indicate low risk, medium scores (30-70) indicate moderate risk requiring monitoring, and high scores (70-100) indicate high risk requiring immediate attention."
    },
    {
      question: "How accurate is the X-ray analysis?",
      answer: "Our AI-powered analysis is trained on thousands of medical images and provides a preliminary assessment. However, it should always be reviewed by qualified medical professionals before any treatment decisions are made."
    },
    {
      question: "What should I do if I have a high risk score?",
      answer: "If you have a high risk score, please schedule an appointment with your doctor immediately. They will review your results and recommend appropriate next steps, which may include additional tests or treatments."
    },
    {
      question: "How often should I get checked?",
      answer: "The frequency of check-ups depends on your individual risk factors and your doctor's recommendations. Generally, high-risk patients may need quarterly check-ups, while low-risk patients may only need annual screenings."
    },
    {
      question: "Can I share my report with other doctors?",
      answer: "Yes, you can download or print your report from the Full Report view. You can also ask your doctor to share it directly through our system."
    },
    {
      question: "What is cancer stratification?",
      answer: "Cancer stratification is the process of categorizing patients based on their risk levels. This helps doctors prioritize care and create personalized treatment plans."
    }
  ];

  return (
    <div className="patient-faq">
      <button className="btn-back" onClick={() => navigate(-1)}>← Back</button>
      
      <div className="faq-header">
        <h1>📚 Frequently Asked Questions</h1>
        <p>Educational information about your reports and cancer screening</p>
      </div>

      <div className="faq-list">
        {faqs.map((faq, index) => (
          <div key={index} className="faq-item">
            <h3>Q: {faq.question}</h3>
            <p>A: {faq.answer}</p>
          </div>
        ))}
      </div>

      <div className="contact-section">
        <h3>Still have questions?</h3>
        <p>Contact your healthcare provider for personalized advice.</p>
      </div>
    </div>
  );
}

export default PatientFAQ;
