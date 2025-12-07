const Joi = require('joi');
const bcrypt = require('bcrypt');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const AuthService = require("../services/auth.service");
const DatabaseService = require("../services/database.service");

// Validation for Partners
const partnerRegisterSchema = Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().min(8).required(),
    firstName: Joi.string().required(),
    lastName: Joi.string().required(),
    companyName: Joi.string().required(),
    phoneNumber: Joi.string().pattern(/^[0-9]{10}$/).required(),
    address: Joi.string().required(),
    country: Joi.string().required()
});

const PartnersController = {

    // --- 1. Register Partner ---
    registerPartner: async (req, res) => {
        try {
            const { error, value } = partnerRegisterSchema.validate(req.body);
            if (error) return res.status(400).send({ error: error.details[0].message });

            const collection = await DatabaseService.goToCollection('Partners');
            const existing = await collection.findOne({ email: value.email });
            if (existing) return res.status(409).send({ message: 'Partner account already exists' });

            const saltRounds = 10;
            const hashedPassword = await bcrypt.hash(value.password, saltRounds);

            const newPartner = {
                ...value,
                password: hashedPassword,
                role: 'partner',
                createdAt: new Date().toISOString()
            };

            await collection.insertOne(newPartner);
            res.status(201).send({ message: 'Partner registered successfully!' });
        } catch (err) {
            console.error("Partner Register Error:", err);
            res.status(500).send({ error: 'Internal Server Error' });
        }
    },

    // --- 2. Login Partner ---
    loginPartner: async (req, res) => {
        try {
            const { email, password } = req.body;
            const collection = await DatabaseService.goToCollection('Partners');

            const partner = await collection.findOne({ email });
            if (!partner) return res.status(401).send('Invalid email or password');

            const isMatch = await bcrypt.compare(password, partner.password);
            if (!isMatch) return res.status(401).send('Invalid email or password');

            const token = AuthService.signJWT(partner);

            res.status(200).json({ 
                message: 'Partner logged in!', 
                token,
                user: { firstName: partner.firstName, company: partner.companyName } 
            });
        } catch (err) {
            console.error("Partner Login Error:", err);
            res.status(500).send({ error: 'Internal Server Error' });
        }
    },

    // --- 3. Get Dashboard Data ---
    getPartnerDashboard: async (req, res) => {
        try {
            const authHeader = req.headers.authorization;
            if (!authHeader) return res.status(401).send({ error: 'No token' });
            
            const token = authHeader.split(' ')[1];
            const decoded = AuthService.verifyJWT(token);
            if (!decoded) return res.status(401).send({ error: 'Invalid token' });

            // Mock Data for Dashboard
            const dashboardData = {
                statistics: {
                    infections_today: 145,
                    active_cases_in_area: 1200,
                    risk_level: "High",
                    trend: "Increasing"
                },
                predictions: {
                    next_week_forecast: "Expected 15% increase in flu cases",
                    advisory: "Stock up on antiviral medication"
                },
                company_info: {
                    name: decoded.firstName + "'s Clinic",
                    status: "Active Partner"
                }
            };

            res.status(200).json(dashboardData);
        } catch (err) {
            console.error("Dashboard Data Error:", err);
            res.status(500).send({ error: 'Internal Server Error' });
        }
    },

    // --- 4. Generate Prediction (THIS WAS MISSING) ---
    generatePrediction: async (req, res) => {
        try {
            const { disease } = req.body;
            if (!disease) return res.status(400).send({ error: "Disease name is required" });

            console.log(`🧠 Generating prediction for: ${disease}`);

            // 1. Fetch Data from DB
            const usersCollection = await DatabaseService.goToCollection('Users');
            const users = await usersCollection.find({}, { projection: { previous_conversations: 1 } }).toArray();

            // 2. Prepare JSONL Data
            let dataset = [];
            users.forEach(user => {
                if (user.previous_conversations) {
                    user.previous_conversations.forEach(report => {
                        // Ensure we have valid data before pushing
                        if (report.report_date && report.final_diagnostic) {
                             dataset.push({
                                date: report.report_date,
                                diagnosis: report.final_diagnostic.most_probable_diagnostic
                            });
                        }
                    });
                }
            });

            if (dataset.length === 0) {
                return res.status(400).send({ error: "Not enough data to generate predictions." });
            }

            // 3. Write to JSONL file
            const dataFilePath = path.join(__dirname, '../data_export.jsonl');
            const fileStream = fs.createWriteStream(dataFilePath);
            dataset.forEach(entry => {
                fileStream.write(JSON.stringify(entry) + '\n');
            });
            fileStream.end();

            // 4. Run Python Script
            // We assume 'model.py' is in the root of the backend folder
            const pythonScriptPath = path.join(__dirname, '../../model.py'); 
            
            console.log(`🚀 Running python script: ${pythonScriptPath}`);
            const pythonProcess = spawn('python3', [pythonScriptPath, dataFilePath, disease]);

            let scriptOutput = "";
            let scriptError = "";

            pythonProcess.stdout.on('data', (data) => {
                scriptOutput += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                scriptError += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code !== 0) {
                    console.error("❌ Python Script Error:", scriptError);
                    return res.status(500).send({ error: "Prediction model failed to run." });
                }

                try {
                    console.log("✅ Python Output:", scriptOutput);
                    const result = JSON.parse(scriptOutput);
                    
                    res.status(200).json({
                        success: true,
                        prediction: result.prediction,
                        plot_image: `data:image/png;base64,${result.image_base64}`
                    });
                } catch (e) {
                    console.error("❌ Failed to parse Python output:", scriptOutput);
                    res.status(500).send({ error: "Invalid model output format." });
                }
            });

        } catch (err) {
            console.error("Prediction Controller Error:", err);
            res.status(500).send({ error: 'Internal Server Error' });
        }
    }
};

module.exports = PartnersController;