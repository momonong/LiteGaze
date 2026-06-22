# 📡 Data Collection Guide: Ubuntu Server & Windows Laptop Client

This guide explains how to set up **LexiGaze** with an **Ubuntu Desktop as the Server** (hosting the databases, model trainer, and fusion logic) and a **Windows Laptop as the Client** (with a webcam to record calibration and reading data).

---

## 🏗️ 1. Setup the Ubuntu Desktop Server

Run these commands on your Ubuntu server terminal to install python dependencies, start the integration web app, and open the ngrok tunnel.

### Step 1.1: Install Dependencies
Open a terminal on Ubuntu and ensure you have Python 3.10+, pip, and python virtual environments installed:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev libgl1-mesa-glx
```
*Note: `libgl1-mesa-glx` is required for OpenCV's face landmark preprocessing to work on Linux.*

### Step 1.2: Clone & Install Python Environment
Navigate to the project root directory on your Ubuntu server:
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies using uv or pip
pip install -r requirements.txt
pip install flask opencv-python-headless numpy torch unigaze-personalization
```
*Note: We use `opencv-python-headless` on the Ubuntu server since it doesn't require GUI dependencies (X11 display) for frame-seeking and landmarker processing.*

### Step 1.3: Launch the Flask Server
Run the Flask server with UTF-8 support:
```bash
python3 -X utf8 run.py
```
*The local server is now serving at `http://localhost:8080`.*

---

## 🌐 2. Expose the Server via ngrok Tunnel

Because you are using another laptop, your server needs a public HTTPS URL so the laptop can access the webcam inside the browser safely.

### Step 2.1: Register an ngrok Account & Token
1. Go to [ngrok.com](https://ngrok.com) and create a free account.
2. Retrieve your **Authtoken** from the ngrok dashboard.

### Step 2.2: Run the Tunnel Script
On a separate terminal tab on your Ubuntu server, run our cross-platform remote collection add-on:
```bash
# Configure your token (one-time action)
python3 scripts/setup_remote_collection.py
```
If it is the first time running:
1. It automatically downloads the official Linux `ngrok` binary, extracts it into `scripts/bin/ngrok`, and sets executable permissions.
2. You will be prompted to run:
   ```bash
   scripts/bin/ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
   ```
3. Re-run the script:
   ```bash
   python3 scripts/setup_remote_collection.py
   ```
4. Copy the public **Calibration URL** (e.g. `https://xxxx.ngrok-free.app/gaze`) or scan the QR Code printed in the terminal.

---

## 💻 3. Collect Gaze Data from Windows Laptop

Now you can sit in front of the Windows Laptop and complete the data collection process.

### Step 3.1: Open the Calibration URL
1. Open Google Chrome or Microsoft Edge on your Windows laptop.
2. Go to the public ngrok link: `https://xxxx.ngrok-free.app/gaze`.
3. Accept webcam permissions when prompted. You should see your live camera feed in the bottom circle.

### Step 3.2: Complete the Calibration & Video Recording
1. Open **收集設定** (Collection Settings) by clicking the gear icon at the bottom right.
2. Set a **受試者 ID** (Participant ID, e.g., `user_laptop_01`).
3. Make sure the checkbox **同時錄製影片** (Record Video) is checked.
4. Click **開始** (Start).
5. Look at the red dot as it moves across the screen. 
6. Once completed:
   * The laptop browser stops the recording and sends both the raw video file and target coordinates to the Ubuntu server.
   * The Ubuntu server processes the frames, creates the calibration dataset, and automatically trains a polynomial regression model.
   * **Local Backup**: The laptop browser will automatically download the recorded video (`<participant>_calibration.webm`) and the timeline JSON (`<participant>_timeline.json`) to your laptop's download folder. If the server fails to process due to network drops, you can manually upload these files to the server later!

---

## 📊 4. Verify & Run Multi-Modal Fusion

1. On your Windows laptop, open `https://xxxx.ngrok-free.app/` (the main reading portal).
2. Upload a text or PDF file and select the trained personalization model (e.g., `user_laptop_01_video_model`) from the dropdown.
3. Toggle the **Live Gaze Tracking** connection and read the document naturally.
4. When finished, click **Analyze & Fuse**. The server will calculate the Reading Difficulty Score (RDS) and display the high-difficulty highlights on your laptop screen.
