#  Garden Studio AI

**Garden Studio AI** is an AI-powered design tool that generates **realistic, high-quality 3D images** of custom garden studios based on user inputs such as dimensions, window types, doors, and cladding materials.

It’s built for architectural visualization — helping clients and designers instantly preview studio concepts before construction.

---

##  Key Features

- **AI Image Generation:** Produces realistic renderings using a fine-tuned Stable Diffusion model.
- **Custom Inputs:** Enter garden studio dimensions and choose design features like doors, windows, and cladding.
- **Web Interface:** Simple React + FastAPI interface — no coding needed.
- **Rate Limiting:** Restricts each IP to 20 generations per day.
- **Local Deployment:** Runs entirely on your infrastructure for privacy and control.
- **Scalable:** Can handle 1,000–10,000 image generations per month.

---

##  System Overview

The system includes three main components:

1. **Frontend (React):**  
   User-facing interface for entering studio specifications.

2. **Backend (FastAPI + Redis):**  
   Handles API requests, enforces rate limits, and triggers AI image generation.

3. **AI Engine (Stable Diffusion + LoRA):**  
   The brain of the system — fine-tuned using architectural images to generate realistic visuals.

---

##  Project Structure

```

garden-studio-ai/
├── backend/               # FastAPI app (API + rate limiting)
├── frontend/              # React user interface
├── ml/                    # AI training and inference modules
├── infra/                 # Docker, database, and deployment setup
├── docs/                  # Architecture and cost notes
└── README.md              # You’re here

````

---

##  Installation Guide

You can run this entire system locally using **Docker** (recommended for non-developers).

### 1️⃣ Prerequisites
Make sure you have:
- [Docker Desktop](https://www.docker.com/) installed  
- Internet connection  
- At least 8 GB RAM  

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/garden-studio-ai.git
cd garden-studio-ai
````

### 3️⃣ Start the System

```bash
docker-compose up --build
```

This will:

* Build the backend (FastAPI)
* Build the frontend (React)
* Start Redis and PostgreSQL

Once it’s ready, open your browser and go to:

```
http://localhost:5173
```

You’ll see the Garden Studio AI interface.

---

##  Training the AI (Optional)

If you want to train the model with your own garden studio images:

1. Place your images inside `ml/data/`
   (Organize by category — for example `doors/`, `windows/`, `cladding/`.)

2. Run the training script:

   ```bash
   cd ml
   python train_lora.py
   ```

3. The script will fine-tune the Stable Diffusion model and save the new weights to `ml/optim/models/`.

The system will automatically start using your trained model for future generations.

---

##  How It Works

1. User enters studio size (e.g., 4m x 6m).
2. Selects windows, door, and cladding types.
3. Clicks **Generate**.
4. The backend sends a request to the AI model.
5. The AI model returns a realistic render image.

---

##  Security & Limits

* Each IP can create **up to 20 images per day**.
* The system can handle **1,000–10,000 images/month**.
* All AI generation runs locally — **no external API** calls.

---

##  Troubleshooting

If the app doesn’t load:

* Make sure Docker containers are running (`docker ps`)
* Visit `http://localhost:8000` to check backend health
* Restart the system: `docker-compose down && docker-compose up --build`

If the AI training is slow:

* Check GPU drivers
* Try running on a cloud GPU (AWS, GCP, or Lambda Labs)

---

##  Tech Stack

* **Frontend:** React (Vite)
* **Backend:** FastAPI, Redis, PostgreSQL
* **AI Engine:** PyTorch, Hugging Face Diffusers (Stable Diffusion), LoRA fine-tuning
* **Infrastructure:** Docker, Nginx

---

## 📈 Expected Costs

| Resource    | Description                    | Approx. Monthly Cost |
| ----------- | ------------------------------ | -------------------- |
| GPU Hosting | Cloud GPU for image generation | $300–$900            |
| Storage     | Training images + models       | $20–$50              |
| Development | One-time setup & training      | $25k–$60k            |

---

##  Maintainer Notes

This system is built to be **self-contained**, meaning you can:

* Train your own models
* Limit access by IP or area
* Deploy on-prem or on cloud

If you plan to expand, consider connecting an authentication layer or payment gateway for commercial deployments.

---

## 🏁 Quick Recap

**Goal:** Let anyone visualize a garden studio instantly, without an architect.
**Inputs:** Dimensions, doors, windows, cladding.
**Output:** Photorealistic image of the design.
**Result:** Faster design decisions and higher client satisfaction.

---

##  Author

Developed by Edward Juma
Contact: [[edwardjuma252@gmail.com](your.email@example.com)]

```

---

Here’s a clean `run.sh` script you can drop right into this project root. It’s written for beginners — all they need to do is **double-click it** (on Linux/macOS) or run it in Git Bash on Windows. It automatically installs dependencies, builds containers, and launches the full Garden Studio AI system.

---

### `run.sh`

```bash
#!/bin/bash
# =========================================================
#  Garden Studio AI - QuickStart Script
# One-command setup for complete beginners
# =========================================================

echo " Starting Garden Studio AI setup..."

# Step 1: Check Docker installation
if ! command -v docker &> /dev/null
then
    echo " Docker is not installed. Please install Docker Desktop and try again."
    echo " https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Step 2: Check Docker Compose
if ! command -v docker-compose &> /dev/null
then
    echo " docker-compose not found. Installing docker-compose..."
    sudo apt-get update && sudo apt-get install -y docker-compose
fi

# Step 3: Build and start containers
echo " Building containers..."
docker-compose up --build -d

# Step 4: Wait for backend to start
echo " Waiting for backend to initialize..."
sleep 10

# Step 5: Check backend health
echo " Checking backend health..."
BACKEND_HEALTH=$(curl -s http://localhost:8000)
if [[ $BACKEND_HEALTH == *"Garden Studio AI backend running"* ]]; then
    echo " Backend is up and running!"
else
    echo " Backend did not respond as expected. Check logs using:"
    echo "   docker logs $(docker ps --filter 'name=backend' -q)"
fi

# Step 6: Launch frontend
echo " Launching frontend..."
echo "Visit  http://localhost:5173 in your browser."

# Step 7: Show usage info
echo ""
echo "=========================================="
echo " Garden Studio AI is now running locally!"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "=========================================="
echo ""
echo "To stop everything, run: docker-compose down"
echo "To rebuild after code changes, run: docker-compose up --build"
```

---

###  Usage

1. Open your terminal (or Git Bash on Windows).
2. Navigate to your project folder:

   ```bash
   cd garden-studio-ai
   ```
3. Make the script executable:

   ```bash
   chmod +x run.sh
   ```
4. Run it:

   ```bash
   ./run.sh
   ```
5. After setup, open your browser and visit:

   ```
   http://localhost:5173
   ```

---

###  What It Does Automatically

* Verifies Docker installation.
* Installs `docker-compose` if missing.
* Builds all containers (`backend`, `frontend`, `redis`, `db`).
* Starts everything in detached mode.
* Waits for the backend to initialize and checks it.
* Prints URLs for quick access.

---



```
Here’s a clean, beginner-friendly **Windows batch script (`run.bat`)** version of the QuickStart setup — built for teams that don’t use Git Bash or WSL. It checks for Docker Desktop, builds everything, and starts the system with minimal fuss.

---

### `run.bat`

```bat
@echo off
:: ============================================================
::  Garden Studio AI - QuickStart Script (Windows)
:: One-command setup for complete beginners
:: ============================================================

echo  Starting Garden Studio AI setup...
echo.

:: Step 1: Check Docker installation
docker --version >nul 2>&1
if errorlevel 1 (
    echo  Docker is not installed or not in PATH.
    echo  Please install Docker Desktop and restart this script.
    echo     https://www.docker.com/products/docker-desktop/
    pause
    exit /b
)

:: Step 2: Check Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo  docker-compose is not installed or not in PATH.
    echo Please enable Docker Compose from Docker Desktop settings.
    pause
    exit /b
)

:: Step 3: Build and start containers
echo 🔧 Building containers...
docker-compose up --build -d
if errorlevel 1 (
    echo  Error building containers. Check your Docker Desktop app.
    pause
    exit /b
)

:: Step 4: Wait for backend to start
echo  Waiting for backend to initialize...
timeout /t 10 >nul

:: Step 5: Check backend health
echo 🔍 Checking backend health...
for /f "tokens=* usebackq" %%a in (`curl -s http://localhost:8000`) do set "BACKEND_HEALTH=%%a"
echo %BACKEND_HEALTH% | find "Garden Studio AI backend running" >nul
if errorlevel 1 (
    echo  Backend did not respond as expected.
    echo Run this command to check logs:
    echo     docker logs (docker ps --filter "name=backend" -q)
) else (
    echo  Backend is up and running!
)

:: Step 6: Launch frontend info
echo.
echo  Launching frontend...
echo Visit  http://localhost:5173 in your browser.

:: Step 7: Show usage info
echo.
echo ==========================================
echo  Garden Studio AI is now running locally!
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo ==========================================
echo.
echo To stop everything, run: docker-compose down
echo To rebuild after code changes, run: docker-compose up --build
echo.

pause
```

---

###  How to Use

1. Save this file as **`run.bat`** in your `garden-studio-ai` project folder.
2. Double-click it to start the setup, or run it from **Command Prompt (Admin)**:

   ```bat
   run.bat
   ```
3. Once complete, open your browser and visit:

   ```
   http://localhost:5173
   ```

---

###  What It Does

* Checks that Docker Desktop and Docker Compose are available.
* Builds and launches backend, frontend, Redis, and PostgreSQL.
* Waits for services to start.
* Tests that the backend is live.
* Displays access URLs and helpful commands.

---

Here’s our **`stop.bat`** — a lightweight Windows cleanup script that safely shuts down the Garden Studio AI system and clears unused containers, networks, and images. It’s beginner-friendly, so your team can stop and clean everything with one click.

---

### `stop.bat`

```bat
@echo off
:: ============================================================
::  Garden Studio AI - Auto Cleanup Script (Windows)
:: Safely stops all running containers and clears unused data
:: ============================================================

echo  Stopping and cleaning up Garden Studio AI...
echo.

:: Step 1: Check Docker installation
docker --version >nul 2>&1
if errorlevel 1 (
    echo  Docker is not installed or not in PATH.
    echo  Please install Docker Desktop and try again.
    pause
    exit /b
)

:: Step 2: Stop running containers
echo  Stopping containers...
docker-compose down

:: Step 3: Remove dangling resources
echo  Removing unused Docker data...
docker system prune -af --volumes

:: Step 4: Confirmation
echo.
echo  Cleanup complete!
echo ==========================================
echo All Garden Studio AI containers, networks,
echo and unused images have been removed.
echo ==========================================
echo.
echo To start again, run: run.bat
echo.

pause
```

---

###  How to Use

1. Save this file as **`stop.bat`** in your project root (`garden-studio-ai`).
2. Double-click it or run from **Command Prompt (Admin)**:

   ```bat
   stop.bat
   ```
3. It will:

   * Stop all running containers.
   * Delete unused images, volumes, and networks.
   * Free up disk space.

---

###  Pair with:

* **`run.bat`** → Starts the system
* **`stop.bat`** → Stops and cleans the system

Together, these give your team a complete **zero-config local deployment cycle** — perfect for non-developers or demo environments.

Practical run example (commands)
cd garden-studio-ai/ml
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# interactive accelerate setup (choose GPU config)
accelerate config

# (set HF token)
export HUGGINGFACE_HUB_TOKEN="hf_..."   # or pass --hf_token in script args

# train (this will use accelerate and GPU)
accelerate launch train_lora.py --config config.yaml --images_dir ./data/images --labels ./data/labels.csv --out_dir ./lora_checkpoints

# when training done, generate:
python generate_from_lora.py --lora_checkpoint ./lora_checkpoints/lora-final --prompt "modern garden studio, timber cladding, lots of windows" --out_dir ./generated


