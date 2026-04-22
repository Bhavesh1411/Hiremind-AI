"""
HireMind AI - Real-Time Webcam Monitoring Module
=================================================
Provides continuous face detection during interview sessions using
streamlit-webrtc for live video streaming + MediaPipe Face Detection.

Capabilities:
  - Auto-start webcam when interview page loads
  - Continuous real-time face count monitoring (every frame)
  - 2-strike violation system (warning → reset)
  - Obstruction / no-face / multi-face detection
  - 2-second stability buffer against false positives

Dependencies:
  - streamlit-webrtc
  - mediapipe
  - opencv-python
  - av (installed with streamlit-webrtc)
"""

import streamlit as st
import numpy as np
import time
import threading
import traceback
_IMPORT_ERROR = None

try:
    import av
    _AV_AVAILABLE = True
except Exception:
    _AV_AVAILABLE = False
    _IMPORT_ERROR = traceback.format_exc()
    class DummyVideoFrame:
        @staticmethod
        def from_ndarray(*args, **kwargs): return None
    class av_dummy:
        VideoFrame = DummyVideoFrame
    av = av_dummy

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception:
    _MP_AVAILABLE = False
    if not _IMPORT_ERROR: _IMPORT_ERROR = traceback.format_exc()

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False
    if not _IMPORT_ERROR: _IMPORT_ERROR = traceback.format_exc()

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    _WEBRTC_AVAILABLE = True
except Exception:
    _WEBRTC_AVAILABLE = False
    _IMPORT_ERROR = traceback.format_exc()
    class VideoProcessorBase:
        pass
    WebRtcMode = None


# ── CSS Variables & Styles ───────────────────────────────────────────────────

def _apply_monitor_styles():
    st.markdown("""
<style>
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }

    .monitor-status {
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 8px;
    }
    .monitor-ok {
        background: var(--monitor-ok-bg, #dcfce7);
        color: var(--monitor-ok-text, #166534);
        border: 1px solid var(--monitor-ok-border, #86efac);
    }
    .monitor-warn {
        background: var(--monitor-warn-bg, #fef3c7);
        color: var(--monitor-warn-text, #92400e);
        border: 1px solid var(--monitor-warn-border, #fcd34d);
    }
    .monitor-danger {
        background: var(--monitor-danger-bg, #fee2e2);
        color: var(--monitor-danger-text, #991b1b);
        border: 1px solid var(--monitor-danger-border, #fca5a5);
    }
</style>
""", unsafe_allow_html=True)


# ── Thread-Safe Face Detection Processor ─────────────────────────────────────

class FaceDetectionProcessor(VideoProcessorBase):
    """
    Processes video frames in a background thread.
    Uses MediaPipe to detect faces and shares the count
    with the main Streamlit thread via a thread-safe lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._face_count = 0
        self._last_seen_time = time.time()  # Track last time a face was seen
        self._last_process_time = 0.0
        self._process_interval = 0.5  # Process every 0.5 seconds (2 FPS detection for efficiency)
        self._buffered_status = "ok"  # Buffered status for UI stability

        if _MP_AVAILABLE and hasattr(mp, "solutions"):
            try:
                self._face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.5,
                )
                # CLAHE for brightness normalization (low light handling)
                self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                self._drawing = mp.solutions.drawing_utils
            except Exception as e:
                logger.error("MediaPipe solution init failed: %s", e)
                self._face_detector = None
                self._drawing = None
        else:
            self._face_detector = None
            self._drawing = None


    @property
    def face_count(self) -> int:
        with self._lock:
            return self._face_count

    @property
    def buffered_status(self) -> str:
        """Returns the status after applying the 2s stability buffer."""
        with self._lock:
            return self._buffered_status

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        """Called for every video frame from the webcam stream."""
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # ── Step 1: Brightness Normalization (CLAHE) ─────────────────────────
        if _CV2_AVAILABLE:
            # Convert to LAB to normalize only the lightness channel
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_norm = self._clahe.apply(l)
            lab_norm = cv2.merge((l_norm, a, b))
            img = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

        # ── Step 2: Periodic Detection ───────────────────────────────────────
        if now - self._last_process_time >= self._process_interval:
            self._last_process_time = now

            if self._face_detector is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self._face_detector.process(rgb)

                count = 0
                if results.detections:
                    count = len(results.detections)
                    self._last_seen_time = now # Update last seen

                    # Draw bounding boxes (thicker for visibility)
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = img.shape
                        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                        x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
                        
                        color = (0, 255, 0) if count == 1 else (0, 0, 255)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                with self._lock:
                    self._face_count = count
                    
                    # ── Buffer Logic ──────────────────────────────────────────
                    # Change buffered status only if face missing for > 2 seconds
                    if count == 1:
                        self._buffered_status = "ok"
                    elif count == 0:
                        if now - self._last_seen_time > 2.0:
                            self._buffered_status = "warning"
                        else:
                            # Maintain "ok" status during temporary flicker
                            self._buffered_status = "ok"
                    else:
                        # Multiple faces detected (instant danger)
                        self._buffered_status = "danger"
            else:
                with self._lock:
                    self._face_count = 1
                    self._buffered_status = "ok"

        # ── Step 3: Visualization Overlay ─────────────────────────────────────
        with self._lock:
            current_count = self._face_count
            status = self._buffered_status

        if status == "ok":
            label, color = "FACE OK", (0, 200, 0)
        elif status == "warning":
            label, color = "NO FACE", (0, 165, 255)
        else:
            label, color = f"{current_count} FACES DETECTED", (0, 0, 255)

        # High-contrast label background
        cv2.rectangle(img, (0, 0), (250, 45), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Identity Capture Processor (Photo-grab with live overlay) ────────────────

class IdentityCaptureProcessor(VideoProcessorBase):
    """
    Video processor for the **identity verification capture** step.

    Draws a real-time face alignment guide directly onto every webcam frame:
      • Rule-of-thirds grid        (faint blue dashed lines)
      • Face alignment oval        (dashed blue ellipse, centred)
      • Corner frame brackets      (registration marks)
      • Centre crosshair           (pin-point aid)
      • "Align face in oval" label (bottom centre with background)

    The latest processed frame is stored thread-safely so the Streamlit main
    thread can snapshot it when the user clicks 'Capture Photo'.
    """

    def __init__(self):
        self._lock          = threading.Lock()
        self._latest_frame  = None   # np.ndarray (BGR)

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")

        if not _CV2_AVAILABLE:
            # No OpenCV — pass frame through unmodified
            with self._lock:
                self._latest_frame = img.copy()
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        h, w = img.shape[:2]

        # ── Rule-of-thirds grid ──────────────────────────────────────────────
        grid_col = (160, 185, 215)   # muted steel-blue (BGR)
        cv2.line(img, (w // 3,     0), (w // 3,     h), grid_col, 1, cv2.LINE_AA)
        cv2.line(img, (2 * w // 3, 0), (2 * w // 3, h), grid_col, 1, cv2.LINE_AA)
        cv2.line(img, (0, h // 3),     (w, h // 3),     grid_col, 1, cv2.LINE_AA)
        cv2.line(img, (0, 2 * h // 3), (w, 2 * h // 3), grid_col, 1, cv2.LINE_AA)

        # ── Face alignment oval ──────────────────────────────────────────────
        cx  = w // 2
        cy  = int(h * 0.46)
        rx  = int(w * 0.22)
        ry  = int(h * 0.37)
        # Shadow (dark) for contrast on any background
        cv2.ellipse(img, (cx, cy), (rx + 2, ry + 2), 0, 0, 360, (30, 30, 30),   2, cv2.LINE_AA)
        # Main oval (bright blue)
        cv2.ellipse(img, (cx, cy), (rx,     ry    ), 0, 0, 360, (240, 160, 60),  2, cv2.LINE_AA)

        # ── Corner brackets ──────────────────────────────────────────────────
        pad  = 14
        blen = max(22, w // 20)
        bcol = (230, 180, 100)
        bthk = 2
        for (x0, y0), (dx, dy) in [
            ((pad,     pad    ), ( 1,  1)),
            ((w - pad, pad    ), (-1,  1)),
            ((pad,     h - pad), ( 1, -1)),
            ((w - pad, h - pad), (-1, -1)),
        ]:
            cv2.line(img, (x0, y0), (x0 + dx * blen, y0           ), bcol, bthk, cv2.LINE_AA)
            cv2.line(img, (x0, y0), (x0,              y0 + dy * blen), bcol, bthk, cv2.LINE_AA)

        # ── Centre crosshair ─────────────────────────────────────────────────
        clen = 12
        ccol = (180, 200, 220)
        cv2.line(img, (cx - clen, h // 2), (cx + clen, h // 2), ccol, 1, cv2.LINE_AA)
        cv2.line(img, (cx, h // 2 - clen), (cx, h // 2 + clen), ccol, 1, cv2.LINE_AA)

        # ── Bottom label ─────────────────────────────────────────────────────
        label      = "Align face in oval"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        fscale     = max(0.42, w / 1400)
        thick      = 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
        lx = (w - tw) // 2
        ly = h - 14
        # Background pill
        cv2.rectangle(img, (lx - 8, ly - th - 5), (lx + tw + 8, ly + 5), (20, 20, 20), -1)
        cv2.putText(img, label, (lx, ly), font, fscale, (230, 180, 100), thick, cv2.LINE_AA)

        # ── Thread-safe frame store ──────────────────────────────────────────
        with self._lock:
            self._latest_frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest_frame_bytes(self) -> bytes | None:
        """
        Returns the most-recent processed frame as JPEG bytes (quality 92),
        or None if no frame has been received yet.
        Called from the Streamlit main thread when user clicks 'Capture Photo'.
        """
        with self._lock:
            frame = self._latest_frame
        if frame is None or not _CV2_AVAILABLE:
            return None
        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes() if success else None


def _init_monitor_state():
    """Initialize webcam monitoring session variables."""
    if "monitor_warnings" not in st.session_state:
        st.session_state["monitor_warnings"] = 0
    if "monitor_last_violation_time" not in st.session_state:
        st.session_state["monitor_last_violation_time"] = 0.0
    if "monitor_status" not in st.session_state:
        st.session_state["monitor_status"] = "waiting"
    if "webcam_started" not in st.session_state:
        st.session_state["webcam_started"] = False


def reset_monitor_state():
    """Resets monitoring state (called on interview reset)."""
    st.session_state["monitor_warnings"] = 0
    st.session_state["monitor_last_violation_time"] = 0.0
    st.session_state["monitor_status"] = "waiting"
    st.session_state["webcam_started"] = False


# ── Main Monitoring Widget ───────────────────────────────────────────────────

def render_webcam_monitor():
    """
    Renders the real-time webcam monitor panel in the interview UI.

    Uses streamlit-webrtc for continuous streaming and MediaPipe for
    per-frame face detection. Violations are tracked and displayed.

    Returns:
        tuple (bool, bool): 
            - monitor_ok: True if interview should continue, False if violations reset it.
            - webcam_active: True if the webcam has been successfully started by the user.
    """
    _init_monitor_state()
    _apply_monitor_styles()

    st.markdown("""
        <div style="margin-bottom: 8px;">
            <span style="font-weight:700; font-size:0.9rem;">📹 Proctoring Monitor</span>
            <span style="color:#94a3b8; font-size:0.78rem;"> — Live webcam required</span>
        </div>
    """, unsafe_allow_html=True)

    if not _WEBRTC_AVAILABLE:
        import sys
        st.error(f"❌ streamlit-webrtc is not found or failed to load in the current environment.")
        st.info(f"**Executable:** `{sys.executable}`")
        if _IMPORT_ERROR:
            with st.expander("🔍 View Technical Error Details"):
                st.code(_IMPORT_ERROR)
        with st.expander("📂 View Search Paths"):
            st.code("\n".join(sys.path))
        st.markdown("Please run: `pip install streamlit-webrtc` in the correct environment.")
        return True, True

    # ── User-Triggered Start ──────────────────────────────────────────────────
    if not st.session_state["webcam_started"]:
        st.info("⚠️ Please allow camera access to continue.")
        if st.button("🚀 Start Interview & Enable Camera", use_container_width=True, type="primary"):
            st.session_state["webcam_started"] = True
            st.rerun()
        return True, False

    # ── WebRTC Streamer (With Resolution Optimization) ────────────────────────
    ctx = webrtc_streamer(
        key="interview_proctoring",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceDetectionProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20}
            }, 
            "audio": False
        },
        async_processing=True,
        desired_playing_state=True,
    )

    # ── Read buffered status from the processor (thread-safe) ─────────────────
    if ctx.video_processor:
        status = ctx.video_processor.buffered_status
        face_count = ctx.video_processor.face_count
        now = time.time()
        
        last_violation = st.session_state["monitor_last_violation_time"]
        BUFFER_SECONDS = 2.0

        if status == "ok":
            st.session_state["monitor_status"] = "ok"
            st.markdown(
                '<div class="monitor-status monitor-ok">✅ Monitoring active — Face detected</div>',
                unsafe_allow_html=True,
            )

        elif status == "warning":
            st.session_state["monitor_status"] = "warning"
            st.markdown(
                '<div class="monitor-status monitor-warn">⚠️ ALERT: No face detected — Return to camera</div>',
                unsafe_allow_html=True,
            )
            # Track violations based on actual buffered state
            if now - last_violation > BUFFER_SECONDS:
                st.session_state["monitor_last_violation_time"] = now
                st.session_state["monitor_warnings"] += 1

        else:
            st.session_state["monitor_status"] = "danger"
            st.markdown(
                f'<div class="monitor-status monitor-danger">🔴 SECURITY ALERT: {face_count} faces detected</div>',
                unsafe_allow_html=True,
            )
            if now - last_violation > BUFFER_SECONDS:
                st.session_state["monitor_last_violation_time"] = now
                st.session_state["monitor_warnings"] += 1

    elif ctx.state.playing:
        st.markdown(
            '<div class="monitor-status monitor-warn">⏳ Initializing face detection...</div>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state["monitor_status"] = "waiting"
        st.markdown(
            '<div class="monitor-status monitor-warn">📷 Permission required — Click START above if requested</div>',
            unsafe_allow_html=True,
        )

    # ── Violation counter display ────────────────────────────────────────────
    warnings = st.session_state["monitor_warnings"]
    st.caption(f"Violations: {warnings}/2")

    # ── Check for session reset trigger ──────────────────────────────────────
    if warnings >= 2:
        st.error(
            "🚫 **Interview Terminated** — Multiple proctoring violations detected. "
            "Your session will be reset."
        )
        return False, True

    elif warnings == 1:
        st.warning(
            "⚠️ **Warning 1/2** — Ensure your face is clearly visible and only one "
            "person is in the frame. Next violation will reset your interview."
        )

    return True, True
