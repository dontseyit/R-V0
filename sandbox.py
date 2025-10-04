from flask import Flask, Response, render_template_string

from command import Command
from src.camera import Detection


# IMPLEMENTATION PLAN
# - Import `Flask`, `Response`, `render_template_string`, `Detection`, and `Command`; create a single Flask app instance plus long-lived `Detection` and `Command` objects to reuse the camera stream and base connection.
# - Mirror the frame generator pattern from `OLD/shadowfax/follow_person_simple.py` using `Detection.get_frame()` for capture and `Detection.detect_object()` for optional annotations; ensure each loop encodes JPEG bytes and yields the multipart payload.
# - Add a `/stream` route that returns `Response(stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")` so the browser can display the MJPEG feed.
# - Create a minimal `index` view that serves an inline HTML template containing an `<img>` tag pointed at `/stream`; include a headline/description if helpful.
# - Tie in `Command` by issuing a neutral `drive(0, 0)` when the generator exits so the base stops if the stream breaks, reusing the helper semantics from `command.py`.
# - Leave the `if __name__ == "__main__"` block to run the Flask dev server with the existing host/port configuration.


app = Flask(__name__)
detection = Detection()
command = Command()


def stream_frames(object_class: str = "person"):
    """Generate an MJPEG stream with optional detection annotations."""
    import cv2  # Local import to avoid unnecessary dependency at import time.

    try:
        while True:
            frame, img = detection.get_frame()
            try:
                detection.detect_object(frame, img, object_class)
            except Exception:
                # Ignore detection failures and continue streaming raw frames.
                pass

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    except GeneratorExit:
        pass
    finally:
        command.drive(0, 0)


@app.route("/")
def index():
    template = """
    <!doctype html>
    <title>R-V0 Droid Stream</title>
    <h1>R-V0 Live Feed</h1>
    <img src="{{ url_for('stream') }}" alt="Live droid feed" />
    """
    return render_template_string(template)


@app.route("/stream")
def stream():
    return Response(stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False, use_reloader=False)
