#!/usr/bin/env python3
"""
drive.py
Robust loader + converter + Udacity simulator server for NVIDIA PilotNet model.
Now with live steering angle display.

Usage:
    python drive.py model.h5 [--speed 20] [--steering-bias 0.0] [--smooth 0.2] [--auto-bias 120]
"""
import argparse
import os
import sys
import h5py
import traceback
import base64
from io import BytesIO
import time

import numpy as np
from PIL import Image
import cv2

import socketio
import eventlet
import eventlet.wsgi
from flask import Flask

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Lambda, Conv2D, Dense, Dropout, Flatten, ELU
from tensorflow.keras.optimizers import Adam

# ---------------- CONFIG ----------------
INPUT_SHAPE = (66, 200, 3)  # NVIDIA PilotNet input shape
DEFAULT_SPEED = 30.0
DEFAULT_SMOOTH = 0.20
DEFAULT_AUTO_BIAS_FRAMES = 120
HARD_SPEED_CAP = 30.0  # hard speed cap

# ---------------- Model definitions -----------------
def nvidia_pilotnet_model(input_shape=INPUT_SHAPE, lr=1e-4):
    """NVIDIA PilotNet model architecture"""
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))

    # Convolutional layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def dense_fallback_model():
    m = Sequential()
    m.add(Dense(50, input_dim=1, activation='sigmoid'))
    m.add(Dense(30, activation='sigmoid'))
    m.add(Dense(10, activation='sigmoid'))
    m.add(Dense(1))
    m.compile(optimizer=Adam(1e-4), loss='mse')
    return m

# ---------------- Preprocessing -----------------
def nvidia_preprocess(image_rgb):
    """
    Preprocessing for NVIDIA PilotNet model
    Matches the training preprocessing
    """
    # Convert PIL Image to numpy array
    img = np.array(image_rgb)

    # Crop to focus on road area (remove sky and car hood)
    # These values match your training preprocessing
    img = img[60:135, :, :]

    # Resize to NVIDIA's standard input size
    img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[0]), interpolation=cv2.INTER_AREA)

    # Convert to YUV color space (as used in NVIDIA's implementation)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # Convert to float32 (model expects this)
    img = img.astype(np.float32)

    return img

# ---------------- Robust loading helpers -----------------
def try_tf_load(path):
    try:
        return load_model(path, compile=False)
    except Exception:
        return None

def try_legacy_keras_load(path):
    try:
        import keras as legacy_keras
        m = legacy_keras.models.load_model(path, compile=False)
        try:
            save_model(m, path.replace(".h5", "_converted_tf.h5"))
        except Exception:
            pass
        return m
    except Exception:
        return None

def load_weights_into_model(weights_h5_path, model, try_by_name=True):
    try:
        model.load_weights(weights_h5_path)
        return True
    except Exception:
        if try_by_name:
            try:
                try:
                    model.load_weights(weights_h5_path, by_name=True, skip_mismatch=True)
                except TypeError:
                    model.load_weights(weights_h5_path, by_name=True)
                return True
            except Exception:
                return False
    return False

def model_file_contains_weights(path):
    try:
        with h5py.File(path, 'r') as f:
            return 'model_weights' in f or 'weights' in f
    except Exception:
        return False

def universal_load(path):
    base = os.path.splitext(path)[0]
    converted_cache = base + "_converted_tf.h5"

    if os.path.exists(converted_cache):
        m = try_tf_load(converted_cache)
        if m is not None:
            return m

    m = try_tf_load(path)
    if m is not None:
        return m

    m = try_legacy_keras_load(path)
    if m is not None:
        return m

    if model_file_contains_weights(path):
        try:
            m_try = nvidia_pilotnet_model()
            ok = load_weights_into_model(path, m_try, try_by_name=True)
            if ok:
                try:
                    save_model(m_try, converted_cache)
                except Exception:
                    pass
                return m_try
        except Exception:
            pass

    dense = dense_fallback_model()
    load_weights_into_model(path, dense, try_by_name=True)
    return dense

# ---------------- Server & inference -----------------
sio = socketio.Server()
app = Flask(__name__)

last_steer = 0.0
smooth_alpha = DEFAULT_SMOOTH
running_bias_mean = 0.0
seen_frames = 0
auto_bias_frames = DEFAULT_AUTO_BIAS_FRAMES
use_auto_bias = auto_bias_frames > 0
steering_static_bias = 0.0
speed_limit = DEFAULT_SPEED

# For tracking and displaying steering info
frame_count = 0
start_time = time.time()
steering_history = []

def ema(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev

def format_steering_display(angle):
    """Format steering angle for visual display"""
    # Create a simple text-based visualization
    bars = 20
    center = bars // 2
    value = int(angle * center + center)
    value = max(0, min(bars-1, value))

    display = [' '] * bars
    display[center] = '|'  # Center marker
    display[value] = '○'   # Current steering position

    # Add direction indicators
    if value < center:
        display[value] = '◄'  # Left turn
    elif value > center:
        display[value] = '►'  # Right turn
    else:
        display[value] = '○'  # Straight

    return ''.join(display)

@sio.on('telemetry')
def telemetry(sid, data):
    global last_steer, running_bias_mean, seen_frames, frame_count, steering_history

    if data:
        try:
            img_string = data["image"]
            img = Image.open(BytesIO(base64.b64decode(img_string))).convert('RGB')
            img = np.asarray(img)

            input_shape = None
            try:
                input_shape = model.input_shape
            except Exception:
                pass

            if input_shape and len(input_shape) == 4:
                proc = nvidia_preprocess(img)
                X = np.expand_dims(proc, axis=0)
            elif input_shape and len(input_shape) == 2:
                dim = input_shape[1]
                if dim == 1:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    feature = np.mean(gray) / 255.0
                    X = np.array([[feature]], dtype=np.float32)
                else:
                    flat = img.astype(np.float32).flatten()
                    required = int(dim)
                    if flat.size >= required:
                        flat = flat[:required]
                    else:
                        pad = np.zeros(required - flat.size, dtype=np.float32)
                        flat = np.concatenate([flat, pad])
                    X = flat.reshape(1, required)
            else:
                proc = nvidia_preprocess(img)
                X = np.expand_dims(proc, axis=0)

            pred = float(model.predict(X, verbose=0)[0][0])

            if use_auto_bias and seen_frames < auto_bias_frames:
                running_bias_mean = (running_bias_mean * seen_frames + pred) / max(1, seen_frames + 1)
                seen_frames += 1
            corrected = pred - (running_bias_mean if use_auto_bias else 0.0)
            corrected += steering_static_bias

            smoothed = ema(last_steer, corrected, smooth_alpha)
            last_steer = smoothed

            speed = float(data.get("speed", 0.0))
            # throttle formula to enforce hard speed cap
            if speed >= HARD_SPEED_CAP:
                throttle = 0.0
            else:
                throttle = 1.0 - (speed / max(1e-6, HARD_SPEED_CAP)) ** 2
                throttle = float(np.clip(throttle, 0.0, 1.0))

            # Update frame count and steering history
            frame_count += 1
            steering_history.append(smoothed)
            if len(steering_history) > 50:  # Keep last 50 values
                steering_history.pop(0)

            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Display steering information
            if frame_count % 5 == 0:  # Update display every 5 frames
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear console

                print("=" * 60)
                print("LIVE STEERING ANGLE DISPLAY")
                print("=" * 60)
                print(f"Frame: {frame_count} | FPS: {fps:.1f} | Speed: {speed:.1f} mph")
                print(f"Steering: {smoothed:7.4f} | Throttle: {throttle:.3f}")
                print()
                print("Steering Visualization:")
                print(format_steering_display(smoothed))
                print("<-- Left        Center        Right -->")
                print()

                # Show recent steering history
                if steering_history:
                    recent_avg = np.mean(steering_history[-10:])  # Last 10 frames
                    recent_std = np.std(steering_history[-10:])
                    print(f"Recent avg: {recent_avg:.4f} | Recent std: {recent_std:.4f}")

                print("=" * 60)

            send_control(smoothed, throttle)
        except Exception as e:
            print("⚠️ Exception in telemetry:", e)
            traceback.print_exc()
            send_control(0.0, 0.0)
    else:
        send_control(0.0, 0.0)

@sio.on('connect')
def connect(sid, environ):
    print("✅ Simulator connected:", sid)
    # Reset tracking variables on new connection
    global frame_count, start_time, steering_history
    frame_count = 0
    start_time = time.time()
    steering_history = []
    send_control(0.0, 0.0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': str(float(steering_angle)),
                            'throttle': str(float(throttle))}, skip_sid=True)

# ---------------- Main -----------------
def main():
    global model, smooth_alpha, auto_bias_frames, use_auto_bias, steering_static_bias, speed_limit

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model .h5 (legacy or tf.keras).")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="target speed (mph)")
    parser.add_argument("--steering-bias", type=float, default=0.0, help="static steering bias added to predictions")
    parser.add_argument("--smooth", type=float, default=DEFAULT_SMOOTH, help="EMA smoothing alpha in [0..1]")
    parser.add_argument("--auto-bias-frames", type=int, default=DEFAULT_AUTO_BIAS_FRAMES, help="frames for auto bias calibration (0 to disable)")
    args = parser.parse_args()

    model_path = args.model
    smooth_alpha = float(np.clip(args.smooth, 0.0, 1.0))
    steering_static_bias = float(args.steering_bias)
    speed_limit = float(args.speed)
    auto_bias_frames = int(max(0, args.auto_bias_frames))
    use_auto_bias = auto_bias_frames > 0

    if not os.path.exists(model_path):
        print("❌ Model file not found:", model_path)
        sys.exit(1)

    print("📂 Loading/Converting model:", model_path)
    try:
        model_loaded = universal_load(model_path)
    except Exception as e:
        print("❌ universal_load failed:", e)
        model_loaded = nvidia_pilotnet_model()

    globals()['model'] = model_loaded
    print("\n📋 Model summary:")
    try:
        model_loaded.summary()
    except Exception:
        pass

    print("🚦 Starting server on port 4567...")
    print("📊 Live steering display will be shown during operation")
    app_ = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app_)

if __name__ == '__main__':
    main()
