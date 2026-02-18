import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from collections import deque
from datetime import datetime
import os
import subprocess
import shutil

# --- CONFIGURATION ---
SESSION_BASE_DIR = 'recording_sessions'
RECORD_BASENAME = 'recording'
LOG_BASENAME = 'detection_log'
BASELINE_REFRESH_FRAMES = 3
BASELINE_SAMPLE_SIZE = 50
BASELINE_HISTORY_FRAMES = 180
BASELINE_PERCENTILE = 75
BASELINE_METHOD = "percentile"  # "ema" or "percentile"
EMA_ALPHA = 0.2
# ---------------------

def start_audio_recording(audio_index, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "avfoundation",
        "-i",
        f":{audio_index}",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        output_path,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def nothing(x):
    pass

def prompt_roi_name(prompt_text, default_value=None):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    result = simpledialog.askstring("ROI Name", prompt_text, initialvalue=default_value, parent=root)
    root.destroy()
    if result is None:
        return None
    return result.strip()

def prompt_choice(title, prompt_text, valid_choices):
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        result = simpledialog.askstring(title, prompt_text, parent=root)
        root.destroy()
        if result is None:
            return None
        result = result.strip()
        if result in valid_choices:
            return result

def prompt_integer(title, prompt_text, default_value=None):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    result = simpledialog.askinteger(title, prompt_text, initialvalue=default_value, parent=root)
    root.destroy()
    return result

def prompt_yes_no(title, prompt_text):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    result = messagebox.askyesno(title, prompt_text, parent=root)
    root.destroy()
    return result

def prompt_video_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(title="Select video file")
    root.destroy()
    return file_path

def show_info(title, text):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showinfo(title, text, parent=root)
    root.destroy()

def draw_controls_panel(frame, is_live):
    controls = [
        "r: draw ROI",
        "b: baseline",
        "d: delete ROI",
        "c: clear all",
        "x: reset board",
        "space: pause",
    ]
    if not is_live:
        controls.extend([
            ".: next frame",
            ",: prev frame",
            "p: prev active",
            "n: next active",
        ])
    else:
        controls.append("R: record")
    controls.append("q: quit")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding_x = 12
    padding_y = 10
    line_gap = 6

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in controls]
    max_width = max((size[0] for size in text_sizes), default=0)
    line_height = max((size[1] for size in text_sizes), default=0)
    panel_width = max_width + padding_x * 2
    panel_height = (line_height + line_gap) * len(controls) + padding_y * 2 - line_gap

    frame_h, frame_w = frame.shape[:2]
    start_x = max(frame_w - panel_width - 10, 0)
    start_y = 10

    cv2.rectangle(frame, (start_x, start_y), (start_x + panel_width, start_y + panel_height), (30, 30, 30), -1)
    cv2.rectangle(frame, (start_x, start_y), (start_x + panel_width, start_y + panel_height), (200, 200, 200), 1)

    cursor_y = start_y + padding_y + line_height
    for line in controls:
        cv2.putText(frame, line, (start_x + padding_x, cursor_y), font, font_scale, (240, 240, 240), thickness)
        cursor_y += line_height + line_gap

# 1. Choose Input Source
print("\n" + "="*50)
print("BUZZER DETECTOR - INPUT SOURCE SELECTION")
print("="*50)
print("1. Use video file")
print("2. Live capture from phone/camera")
print("="*50)

input_source_menu = (
    "BUZZER DETECTOR - INPUT SOURCE SELECTION\n"
    "1. Use video file\n"
    "2. Live capture from phone/camera"
)

while True:
    choice = prompt_choice("Input Source", f"{input_source_menu}\n\nEnter your choice (1 or 2):", ["1", "2"])
    if choice is None:
        raise SystemExit(0)
    break

is_live_capture = (choice == '2')

if is_live_capture:
    print("\n--- LIVE CAPTURE MODE ---")
    print("Connecting to camera...")
    print("Note: Make sure your phone is connected via USB or using an app like DroidCam/iVCam")
    device_id = prompt_integer("Camera Device", "Enter camera device ID (default 0):", 0)
    if device_id is None:
        raise SystemExit(0)
    cap = cv2.VideoCapture(device_id)
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SESSION_BASE_DIR, f"session_{session_stamp}")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Session folder: {session_dir}")
else:
    print("\n--- VIDEO PLAYBACK MODE ---")
    video_path = prompt_video_file()
    if not video_path:
        raise SystemExit(0)
    while not os.path.exists(video_path):
        show_info("File Not Found", f"File not found: {video_path}")
        video_path = prompt_video_file()
        if not video_path:
            raise SystemExit(0)
    cap = cv2.VideoCapture(video_path)
    print(f"Loading video: {video_path}")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video/camera.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:  # Live capture might return 0
    fps = 60  # Default FPS for live capture
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 2. Create Window & Trackbars
cv2.namedWindow('Calibration Tool')
cv2.createTrackbar('Delta', 'Calibration Tool', 30, 50, nothing)

# Only create Frame seekbar for video playback, not live capture
if not is_live_capture:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar('Frame', 'Calibration Tool', 0, max(total_frames - 1, 0), nothing)
else:
    total_frames = 0  # N/A for live capture

rois = []  # Will store tuples: (x, y, w, h, name, was_active, baseline, samples, history, active_history)
last_active_frames = {}
paused = False
recording = False
video_writer = None
active_frames = []  # List to store frame numbers where any ROI became active
log_entries = []  # Store log entries for the current recording
frame_counter = 0  # For live capture, we'll track our own frame count
frames_since_baseline_refresh = 0
recording_index = 0
current_log_path = None
current_video_path = None
current_video_temp_path = None
current_audio_path = None
audio_process = None
audio_enabled = False
audio_device_index = None

if is_live_capture:
    ffmpeg_available = shutil.which("ffmpeg") is not None
    if ffmpeg_available:
        use_audio = prompt_yes_no("Audio Recording", "Enable audio recording for live capture?")
        if use_audio:
            show_info("Audio Devices", "To list audio devices on macOS, run:\nffmpeg -f avfoundation -list_devices true -i \"\"")
            audio_device_index = prompt_integer("Audio Device", "Enter audio device index (e.g., 0 or 1):")
            if audio_device_index is not None:
                audio_enabled = True
    else:
        print("ffmpeg not found. Audio recording disabled.")

print("\n" + "="*50)
print("--- CONTROLS ---")
print("'r'     : Draw a new ROI (Drag mouse, press ENTER)")
print("'b'     : Capture baseline (light OFF) for all ROIs")
print("'c'     : Clear all ROIs")
print("'d'     : Delete ROI by name")
print("'x'     : Reset last-active board")
print("SPACE   : Pause / Play")
if not is_live_capture:
    print("'>' (.) : Next Frame (only when paused)")
    print("'<' (,) : Previous Frame (only when paused)")
    print("'p'     : Jump to Previous Active Frame")
    print("'n'     : Jump to Next Active Frame")
if is_live_capture:
    print("'R'     : Start/Stop Recording (output + log)")
print("'q'     : Quit")
print("="*50)
if not is_live_capture:
    print(f"Video: {total_frames} frames @ {fps} FPS")
else:
    print(f"Live Capture @ {fps} FPS (estimated)")
print("="*50 + "\n")
def restore_baselines_for_frame(target_frame, rois_list):
    for i, roi_data in enumerate(rois_list):
        x, y, w, h, name, was_active, baseline, samples, history, active_history = roi_data
        restored_state = None
        if target_frame in active_history:
            restored_state = active_history[target_frame]
        else:
            for frame_id, frame_baseline, frame_samples in reversed(history):
                if frame_id <= target_frame:
                    restored_state = (frame_baseline, frame_samples)
                    break
        if restored_state is not None:
            baseline, samples = restored_state[0], list(restored_state[1])
        rois_list[i] = (x, y, w, h, name, was_active, baseline, samples, history, active_history)

def draw_last_active_board(frame, rois_list, last_active_map):
    if not rois_list:
        return

    names = [roi[4] for roi in rois_list]
    values = [str(last_active_map.get(name, "-")) for name in names]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding_x = 14
    padding_y = 10
    cell_heights = []
    cell_widths = []

    for name, value in zip(names, values):
        name_size = cv2.getTextSize(name, font, font_scale, thickness)[0]
        value_size = cv2.getTextSize(value, font, font_scale, thickness)[0]
        cell_widths.append(max(name_size[0], value_size[0]) + padding_x * 2)
        cell_heights.append(max(name_size[1], value_size[1]) + padding_y * 2)

    row_height = max(cell_heights) if cell_heights else 0
    total_width = sum(cell_widths)
    total_height = row_height * 2
    start_x, start_y = 10, 10

    # Background board
    cv2.rectangle(frame, (start_x, start_y), (start_x + total_width, start_y + total_height), (40, 40, 40), -1)
    cv2.rectangle(frame, (start_x, start_y), (start_x + total_width, start_y + total_height), (200, 200, 200), 1)
    cv2.line(frame, (start_x, start_y + row_height), (start_x + total_width, start_y + row_height), (200, 200, 200), 1)

    cursor_x = start_x
    for idx, (name, value) in enumerate(zip(names, values)):
        cell_width = cell_widths[idx]
        if idx > 0:
            cv2.line(frame, (cursor_x, start_y), (cursor_x, start_y + total_height), (200, 200, 200), 1)

        name_size = cv2.getTextSize(name, font, font_scale, thickness)[0]
        value_size = cv2.getTextSize(value, font, font_scale, thickness)[0]

        name_x = cursor_x + (cell_width - name_size[0]) // 2
        name_y = start_y + row_height - padding_y
        value_x = cursor_x + (cell_width - value_size[0]) // 2
        value_y = start_y + total_height - padding_y

        cv2.putText(frame, name, (name_x, name_y), font, font_scale, (240, 240, 240), thickness)
        cv2.putText(frame, value, (value_x, value_y), font, font_scale, (0, 255, 255), thickness)

        cursor_x += cell_width

def get_video_frame_index(capture):
    return max(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1, 0)


while True:
    # Handle video playback vs live capture
    frame_advanced = False
    if is_live_capture:
        # Live capture mode - continuous feed
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            frame_counter += 1
            frame_advanced = True
        current_frame = frame_counter
    else:
        # Video playback mode - with seekbar
        seekbar_pos = cv2.getTrackbarPos('Frame', 'Calibration Tool')
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # If seekbar was moved by user, seek to that frame
        if abs(seekbar_pos - current_frame) > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, seekbar_pos)
            ret, frame = cap.read()
            paused = True  # Auto-pause when seeking
            frame_advanced = True
            current_frame = get_video_frame_index(cap)
            cv2.setTrackbarPos('Frame', 'Calibration Tool', current_frame)
            restore_baselines_for_frame(seekbar_pos, rois)
        elif not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop video automatically for convenience
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_advanced = True
            # Update seekbar to current position
            current_frame = get_video_frame_index(cap)
            cv2.setTrackbarPos('Frame', 'Calibration Tool', current_frame)
    
    # Work on a copy of the frame to avoid drawing permanently on the original data
    display_frame = frame.copy()
    
    # Get current delta threshold from slider
    threshold_value = cv2.getTrackbarPos('Delta', 'Calibration Tool')
    # Fast normalization to keep thresholds consistent across lighting changes
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray_frame = cv2.equalizeHist(gray_frame)
    if not is_live_capture and frame_advanced:
        current_frame = get_video_frame_index(cap)
    
    if BASELINE_METHOD != "ema":
        if frame_advanced:
            frames_since_baseline_refresh += 1
        refresh_baseline_now = frames_since_baseline_refresh >= BASELINE_REFRESH_FRAMES
    else:
        refresh_baseline_now = False

    # Track if any ROI just became active in this frame
    any_roi_became_active = False
    
    # Process ROIs
    for i, roi_data in enumerate(rois):
        x, y, w, h, name, was_active, baseline, samples, history, active_history = roi_data
        
        # Extract the region of interest from preprocessed gray frame
        gray_roi = gray_frame[y:y+h, x:x+w]
        avg_brightness = int(np.mean(gray_roi))

        # Initialize baseline for EMA if needed
        if baseline is None and frame_advanced:
            baseline = avg_brightness

        # DECISION LOGIC (baseline-delta thresholding)
        if baseline is None:
            is_active = False
            delta_value = 0
        else:
            delta_value = avg_brightness - baseline
            is_active = delta_value >= threshold_value

        # Collect inactive samples for baseline voting window
        if frame_advanced and is_active is False:
            samples.append(avg_brightness)
            if len(samples) > BASELINE_SAMPLE_SIZE:
                samples.pop(0)

        # Update baseline using EMA or percentile voting
        if BASELINE_METHOD == "ema" and frame_advanced and is_active is False and baseline is not None:
            baseline = int(EMA_ALPHA * avg_brightness + (1 - EMA_ALPHA) * baseline)
            delta_value = avg_brightness - baseline
        elif refresh_baseline_now and samples:
            baseline = int(np.percentile(samples, BASELINE_PERCENTILE))
            delta_value = avg_brightness - baseline
        
        if was_active != is_active and is_active:
            any_roi_became_active = True
        
        # THRESHOLD CROSSING DETECTION
        if was_active != is_active:
            status = "ACTIVE" if is_active else "INACTIVE"
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_msg = f"[{timestamp}] Frame {current_frame}: {name} -> {status} (B:{avg_brightness}, T:{threshold_value})"
            print(f"[THRESHOLD CROSSED] {log_msg}")
            log_entries.append(log_msg)
            if is_active:
                last_active_frames[name] = current_frame
        # Persist baseline history for restores
        if baseline is not None:
            history.append((current_frame, baseline, list(samples)))
            #only store the history of the frame where it was activated and not all active frames
            if is_active and was_active!=is_active: 
                active_history[current_frame] = (baseline, list(samples))
        
        # VISUALIZATION
        # Color: Green (Off) vs Red (On)
        color = (0, 0, 255) if is_active else (0, 255, 0)
        thickness = 3 if is_active else 2
        
        # Draw box
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, thickness)
        
        # Draw Data (Name, Brightness Value / Threshold)
        # This is CRITICAL: Watch these numbers to tune your system
        if baseline is None:
            text = f"{name} - B:{avg_brightness} | D:{threshold_value} | BL:N/A"
        else:
            text = f"{name} - B:{avg_brightness} | BL:{baseline} | D:{delta_value} | DT:{threshold_value}"
        cv2.putText(display_frame, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update the state in the list
        rois[i] = (x, y, w, h, name, is_active, baseline, samples, history, active_history)
    
    if refresh_baseline_now:
        frames_since_baseline_refresh = 0

    # Track active frames
    if any_roi_became_active and current_frame not in active_frames:
        active_frames.append(current_frame)
        active_frames.sort()
    
    # Add status bar at the bottom
    if is_live_capture:
        status_text = f"Live | Frame: {current_frame}"
    else:
        status_text = f"Video | Frame: {current_frame}/{total_frames}"
    
    if recording:
        status_text += " | REC"
    if paused:
        status_text += " | PAUSED"
    
    cv2.putText(display_frame, status_text, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    draw_last_active_board(display_frame, rois, last_active_frames)
    draw_controls_panel(display_frame, is_live_capture)
    
    # Write frame if recording
    if recording and video_writer is not None:
        video_writer.write(display_frame)

    cv2.imshow('Calibration Tool', display_frame)

    # Key Handling
    key = cv2.waitKey(30) & 0xFF  # 30ms delay ~ 30 FPS playback speed

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('.') and paused and not is_live_capture: # Next frame - video only
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos('Frame', 'Calibration Tool', current_frame)
            restore_baselines_for_frame(current_frame, rois)
    elif key == ord(',') and paused and not is_live_capture: # Previous frame - video only
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Go back 2 frames (1 to undo the current position, 1 to go back)
        if current_frame >= 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 2)
            ret, frame = cap.read()
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos('Frame', 'Calibration Tool', current_frame)
            restore_baselines_for_frame(current_frame, rois)
        else:
            print("Already at the beginning of the video")
    elif key == ord('p') and not is_live_capture:  # Jump to previous active frame - video only
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        prev_active = [f for f in active_frames if f < current_frame]
        if prev_active:
            target_frame = prev_active[-1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cv2.setTrackbarPos('Frame', 'Calibration Tool', target_frame)
            paused = True
            restore_baselines_for_frame(target_frame, rois)
            print(f"Jumped to previous active frame: {target_frame}")
        else:
            print("No previous active frames found")
    elif key == ord('n') and not is_live_capture:  # Jump to next active frame - video only
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        next_active = [f for f in active_frames if f > current_frame]
        if next_active:
            target_frame = next_active[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cv2.setTrackbarPos('Frame', 'Calibration Tool', target_frame)
            paused = True
            restore_baselines_for_frame(target_frame, rois)
            print(f"Jumped to next active frame: {target_frame}")
        else:
            print("No next active frames found")
    elif key == ord('R') and is_live_capture:  # Recording - live capture only
        if not recording:
            # Start recording
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            recording_index += 1
            recording_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{RECORD_BASENAME}_{recording_index:03d}_{recording_stamp}.mp4"
            video_temp_filename = f"{RECORD_BASENAME}_{recording_index:03d}_{recording_stamp}_video.mp4"
            log_filename = f"{LOG_BASENAME}_{recording_index:03d}_{recording_stamp}.txt"
            audio_filename = f"{RECORD_BASENAME}_{recording_index:03d}_{recording_stamp}.m4a"
            current_video_path = os.path.join(session_dir, video_filename)
            current_video_temp_path = os.path.join(session_dir, video_temp_filename)
            current_log_path = os.path.join(session_dir, log_filename)
            current_audio_path = os.path.join(session_dir, audio_filename)
            video_writer = cv2.VideoWriter(current_video_temp_path, fourcc, fps, (width, height))
            recording = True
            log_entries = []
            if audio_enabled:
                audio_process = start_audio_recording(audio_device_index, current_audio_path)
                print("Audio recording started")
            print(f"Recording started -> {current_video_path}")
        else:
            # Stop recording
            recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            if audio_process is not None:
                audio_process.terminate()
                audio_process.wait(timeout=3)
                audio_process = None
            # Save log file
            with open(current_log_path, 'w') as f:
                f.write(f"Detection Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                for entry in log_entries:
                    f.write(entry + "\n")
            if audio_enabled and current_audio_path and current_video_temp_path:
                mux_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    current_video_temp_path,
                    "-i",
                    current_audio_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    current_video_path,
                ]
                result = subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode != 0:
                    print("Warning: Failed to mux audio with video. Keeping video-only file.")
                    current_video_path = current_video_temp_path
                else:
                    os.remove(current_video_temp_path)
                    os.remove(current_audio_path)
            else:
                current_video_path = current_video_temp_path
            print(f"Recording stopped. Video saved to {current_video_path}")
            print(f"Log saved to {current_log_path}")
    elif key == ord('r'):
        # Pause video to draw
        paused = True
        # Select ROI
        rect = cv2.selectROI("Calibration Tool", display_frame, fromCenter=False, showCrosshair=True)
        # Handle if user cancels selection (w=0 or h=0)
        if rect[2] > 0 and rect[3] > 0:
            # Ask for a name for this ROI
            roi_name = prompt_roi_name(
                f"Enter name for ROI #{len(rois) + 1} (or leave blank for default):",
                f"ROI_{len(rois) + 1}",
            )
            if roi_name is None:
                print("ROI naming cancelled")
                continue
            if not roi_name:
                roi_name = f"ROI_{len(rois) + 1}"
            # Add ROI with name and initial state (False = inactive)
            history = deque(maxlen=BASELINE_HISTORY_FRAMES)
            active_history = {}
            rois.append((rect[0], rect[1], rect[2], rect[3], roi_name, False, None, [], history, active_history))
            last_active_frames[roi_name] = "-"
            print(f"Added region: {roi_name}")
    elif key == ord('b'):
        # Capture baseline for all ROIs (use current frame)
        if not rois:
            print("No ROIs available to baseline")
        else:
            for i, roi_data in enumerate(rois):
                x, y, w, h, name, was_active, _baseline, samples, history, active_history = roi_data
                gray_roi = gray_frame[y:y+h, x:x+w]
                baseline = int(np.mean(gray_roi))
                samples = [baseline]
                rois[i] = (x, y, w, h, name, False, baseline, samples, history, active_history)
            print("Baseline captured for all ROIs")
    elif key == ord('x'):
        for roi_data in rois:
            last_active_frames[roi_data[4]] = "-"
        print("Last-active board reset")
    elif key == ord('d'):
        if not rois:
            print("No ROIs available to delete")
        else:
            roi_name = prompt_roi_name("Enter ROI name to delete:")
            if roi_name is None:
                print("ROI delete cancelled")
                continue
            if not roi_name:
                print("No ROI name entered")
            else:
                original_count = len(rois)
                rois = [roi_data for roi_data in rois if roi_data[4] != roi_name]
                if len(rois) == original_count:
                    print(f"ROI '{roi_name}' not found")
                else:
                    last_active_frames.pop(roi_name, None)
                    print(f"Deleted ROI '{roi_name}'")
    elif key == ord('c'):
        rois = []
        last_active_frames = {}
        print("All ROIs cleared")

# Cleanup
if recording and current_log_path and log_entries:
    with open(current_log_path, 'w') as f:
        f.write(f"Detection Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for entry in log_entries:
            f.write(entry + "\n")
if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()