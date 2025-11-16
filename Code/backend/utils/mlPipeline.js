import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { parse } from 'csv-parse/sync';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Run the Python ML pipeline on an uploaded image
 * @param {string} imagePath - Full path to the uploaded image
 * @returns {Promise<Object>} - Parsed ML results
 */
export async function runMLPipeline(imagePath) {
  return new Promise((resolve, reject) => {
    console.log(`\n${'='.repeat(80)}`);
    console.log('RUNNING ML RISK ASSESSMENT PIPELINE');
    console.log(`${'='.repeat(80)}`);
    console.log(`Image path: ${imagePath}`);
    
    // Path to the Python script
    const pythonScript = path.join(__dirname, '../risk_model.py');
    
    // Check if Python script exists
    fs.access(pythonScript)
      .catch(() => {
        throw new Error(`Python script not found: ${pythonScript}`);
      });
    
    // Spawn Python process
    const pythonProcess = spawn('python3', [pythonScript, imagePath]);
    
    let stdout = '';
    let stderr = '';
    
    // Collect stdout
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      stdout += output;
      console.log(output.trim());
    });
    
    // Collect stderr
    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString();
      stderr += error;
      console.error('Python Error:', error.trim());
    });
    
    // Handle process completion
    pythonProcess.on('close', async (code) => {
      console.log(`\nPython process exited with code ${code}`);
      console.log(`${'='.repeat(80)}\n`);
      
      if (code !== 0) {
        return reject(new Error(`ML pipeline failed with code ${code}: ${stderr}`));
      }
      
      try {
        // Find the generated CSV file in the backend root directory
        const backendDir = path.join(__dirname, '..');
        const csvFiles = await fs.readdir(backendDir);
        const csvFile = csvFiles
          .filter(f => f.startsWith('risk_assessment_') && f.endsWith('.csv'))
          .sort()
          .pop(); // Get the most recent
        
        if (!csvFile) {
          return reject(new Error('ML pipeline completed but no CSV output found'));
        }
        
        const csvPath = path.join(backendDir, csvFile);
        console.log(`Reading ML results from: ${csvPath}`);
        
        // Read and parse CSV
        const csvContent = await fs.readFile(csvPath, 'utf-8');
        const records = parse(csvContent, {
          columns: true,
          skip_empty_lines: true
        });
        
        if (records.length === 0) {
          return reject(new Error('CSV file is empty'));
        }
        
        const result = records[0];
        
        // Parse and structure the results
        const mlResults = {
          riskScore: parseFloat(result['Risk_Score_%']),
          riskCategory: result['Risk_Category'],
          chexpertScore: parseFloat(result['CheXpert_Score_%']),
          biobertScore: parseFloat(result['BioBERT_Score_%']),
          xgboostScore: parseFloat(result['XGBoost_Score_%']),
          clinicalScore: parseFloat(result['Clinical_Score_%']),
          positiveFindings: result['Positive_Findings'],
          summary: result['Medical_Summary'],
          timestamp: result['Timestamp']
        };
        
        console.log('\n✓ ML Results parsed successfully:');
        console.log(`  Risk Score: ${mlResults.riskScore}%`);
        console.log(`  Risk Category: ${mlResults.riskCategory}`);
        console.log(`  Summary: ${mlResults.summary}`);
        
        // Clean up CSV file
        await fs.unlink(csvPath).catch(err => 
          console.warn(`Could not delete CSV file: ${err.message}`)
        );
        
        resolve(mlResults);
        
      } catch (error) {
        console.error('Error parsing ML results:', error);
        reject(new Error(`Failed to parse ML results: ${error.message}`));
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python process:', error);
      reject(new Error(`Failed to start ML pipeline: ${error.message}`));
    });
  });
}

/**
 * Generate recommended next steps based on risk category and findings
 * @param {string} riskCategory - LOW, MEDIUM, or HIGH
 * @param {string} positiveFindings - Comma-separated findings
 * @returns {string} - Recommended next steps
 */
export function generateRecommendations(riskCategory, positiveFindings) {
  const recommendations = {
    HIGH: [
      'Immediate clinical correlation and follow-up required',
      'Consult with oncology specialist within 24-48 hours',
      'Consider additional diagnostic imaging (CT/MRI)',
      'Biopsy may be recommended',
      'Close monitoring and treatment planning needed'
    ],
    MEDIUM: [
      'Clinical follow-up recommended within 1-2 weeks',
      'Additional imaging may be considered',
      'Monitor symptoms and progression',
      'Regular follow-up appointments scheduled',
      'Lifestyle modifications and preventive care advised'
    ],
    LOW: [
      'Routine monitoring recommended',
      'Follow-up imaging in 6-12 months',
      'Maintain healthy lifestyle',
      'Report any new symptoms promptly',
      'Annual health screening advised'
    ]
  };
  
  const baseRecommendations = recommendations[riskCategory] || recommendations.MEDIUM;
  
  // Add specific recommendations based on findings
  const findingsLower = positiveFindings.toLowerCase();
  const specificRecommendations = [];
  
  if (findingsLower.includes('pneumonia')) {
    specificRecommendations.push('Antibiotic therapy may be indicated');
  }
  if (findingsLower.includes('pneumothorax')) {
    specificRecommendations.push('Immediate chest tube placement may be required');
  }
  if (findingsLower.includes('effusion')) {
    specificRecommendations.push('Thoracentesis may be considered for large effusions');
  }
  if (findingsLower.includes('lesion') || findingsLower.includes('mass')) {
    specificRecommendations.push('Tissue diagnosis recommended');
  }
  
  return [...baseRecommendations, ...specificRecommendations].join('. ') + '.';
}
