from website import create_app
from flask_socketio import SocketIO, emit
from website.recognition import recognize_face
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)
logging.getLogger('socketio').setLevel(logging.WARNING)


app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Handle incoming webcam frames from the proctoring engine.
    Runs InsightFace for face identity verification and returns:
    - name: closest match from embeddings
    - confidence: similarity score
    - face_count: number of faces detected
    """
    if data:
        try:
            result = recognize_face(data)
            emit('response', result)
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            emit('response', {'name': None, 'confidence': None, 'face_count': 0, 'error': str(e)})
    else:
        emit('response', {'name': None, 'confidence': None, 'face_count': 0, 'error': 'No image data received'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5100, debug=True)
