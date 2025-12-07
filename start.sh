#!/bin/bash

# 1. Python Server (Assumes script is run from root)
gnome-terminal --tab --title="Python API" -- bash -c "uvicorn server:app --reload --host 0.0.0.0 --port 8000; exec bash"

# 2. Node Server
gnome-terminal --tab --title="Node Backend" -- bash -c "cd backend/sourceCode && node server.js; exec bash"

# 3. Angular Frontend
gnome-terminal --tab --title="Angular Frontend" -- bash -c "cd frontend && ng serve; exec bash"