// backend/server.js
const express = require('express');
const cors = require('cors');
const app = express();
const port = 3000;

// Enable CORS to allow requests from Angular (running on port 4200)
app.use(cors());
app.use(express.json());

// A simple API route
app.get('/api/message', (req, res) => {
    res.json({ message: 'Hello from the Node.js Backend!' });
});

// Start the server
app.listen(port, () => {
    console.log(`Backend server listening at http://localhost:${port}`);
});