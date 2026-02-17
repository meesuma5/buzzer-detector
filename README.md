# Buzzer Detector

A realtime computer-vision tool for instantly detecting a buzzer/light turning on in video or live camera feeds. It provides ROI-based calibration, thresholding, and frame navigation to quickly validate detections and tune settings.

## Project Overview
This repository contains a single OpenCV-based script that:
- Lets you draw named regions of interest (ROIs).
- Computes brightness changes and detects activation events in realtime.
- Logs threshold crossings for later review.
- Supports both video playback and live capture modes.
- Optionally records annotated video and audio (live mode).

## What It Does
- **ROI-based detection:** Track only the buzzer area instead of the whole frame.
- **Rolling baseline:** Baselines are computed from recent inactive frames and refreshed on a timer.
- **Active-frame navigation:** Jump through detected activity frames with shortcuts.
- **Session artifacts:** Logs and recordings are stored per session.

## Engineering Depth (Why This Is Nontrivial)
- **Frame accumulation and state:** Each ROI maintains a rolling history of baseline snapshots to keep detection stable over time.
- **Inactive-only sampling:** Baseline updates are computed only from frames classified as inactive to avoid contamination.
- **Timeline recovery:** When jumping across the timeline, baseline state is restored for the target frame to preserve consistency.
- **Realtime + playback parity:** The same detection pipeline supports both live capture and offline analysis.
- **Integrated session output:** Video, audio, and logs are synchronized and stored per session for auditability.

## System Snapshot

![System in action](assets/system-demo.png)

## Requirements
- Python 3.9+
- OpenCV (cv2)
- NumPy
- ffmpeg (optional, for live audio recording on macOS)

Install dependencies:
```bash
pip install -r requirements.txt
```

## How To Use
Run the script:
```bash
python mock.py
```
You will be prompted to choose between video playback and live capture.

### Video Playback Mode
1. Choose option `1` and enter your video filename.
2. Draw ROIs with `r` and name them.
3. Adjust the `Delta` slider until activation matches the buzzer turning on.
4. Use frame navigation:
   - `.` / `,` step forward/back when paused
   - `p` / `n` jump to previous/next active frame

### Live Capture Mode
1. Choose option `2` and select your camera device ID.
2. (Optional) Enable audio recording if `ffmpeg` is available.
3. Draw ROIs and tune the `Delta` slider.
4. Use `R` to start/stop recording. Output files and logs go to a session folder.

## Controls
- `r` : Draw a new ROI
- `c` : Clear all ROIs
- `SPACE` : Pause/Play
- `.` : Next frame (paused, video mode)
- `,` : Previous frame (paused, video mode)
- `p` : Previous active frame (video mode)
- `n` : Next active frame (video mode)
- `R` : Start/stop recording (live mode)
- `q` : Quit

## Output
- **Logs:** Threshold crossing events are written to a timestamped log file.
- **Recordings:** Saved to the session folder (live capture mode).

## Notes
- Baseline is updated using recent inactive frames only.
- Jumping with `p`/`n` restores the baseline state for that frame.
