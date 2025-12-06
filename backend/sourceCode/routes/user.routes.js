const express = require('express');
const router = express.Router();

const UserController = require('../controllers/user.controller')

router.post('/login', UserController.loginUser);
router.post('/register', UserController.registerUser);
router.get('/profile', UserController.getUserProfile); 

module.exports = router;