import os
import base64
import pickle

import numpy as np
import cv2
from flask import Blueprint, request, jsonify, session

recognition = Blueprint('recognition', __name__)

_PKL_PATH = os.path.join(os.path.dirname(__file__), 'embeddings.pkl')

# Lazy-loaded globals — only initialised on first call to recognize_face()
_detector = None
_embeddings = None
_names = None
_temp_embeddings = {}  # Thread-safe global store: regno -> face embedding numpy array


def compute_cosine_similarity(a, b):
    """
    Pure NumPy Cosine Similarity calculation.
    Avoids heavy SciPy / Fortran LAPACK DLL loads that fail on Windows memory paging.
    Formula: (A . B^T) / (||A|| * ||B||)
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    norm_a[norm_a == 0] = 1e-10
    norm_b[norm_b == 0] = 1e-10
    return np.dot(a / norm_a, (b / norm_b).T)


def _load_model():
    """Load InsightFace model and embeddings on first use."""
    global _detector, _embeddings, _names
    if _detector is not None:
        return  # already loaded

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise RuntimeError(
            "insightface is not installed in this Python environment.\n"
            "Activate the project venv first:\n"
            "  Windows:  venv\\Scripts\\activate\n"
            "  Mac/Linux: source venv/bin/activate\n"
            "Then: pip install insightface"
        )

    with open(_PKL_PATH, 'rb') as f:
        _embeddings, _names = pickle.load(f)
    print(f"Loaded embeddings shape: {_embeddings.shape}, names: {_names}")

    _detector = FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'recognition'])
    _detector.prepare(ctx_id=-1, det_size=(640, 640))


def register_temp_face(regno, image_data):
    """
    Decode a base64 JPEG, detect a single face, extract its normalized embedding,
    and save it to the global temporary storage dict.
    Returns: (success_bool, message_str)
    """
    _load_model()
    try:
        np_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return False, f"Invalid camera image format: {str(e)}"

    if img is None:
        return False, "Failed to decode camera snapshot."

    # Keep image dimensions compact to prevent memory spikes
    h, w = img.shape[:2]
    if max(h, w) > 480:
        scale = 480.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        faces = _detector.get(img)
    except Exception as e:
        print(f"InsightFace detector error: {e}")
        return False, "Face analysis model failed to process the image. Please try again."

    if not faces:
        return False, "No face detected. Please ensure your face is fully visible in front of the camera."

    if len(faces) > 1:
        return False, "Multiple faces detected. Please make sure you are alone in the frame."

    # Process primary face
    face = faces[0]
    embedding = face.normed_embedding
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # Save to temporary in-memory store
    _temp_embeddings[regno] = embedding
    print(f"[Face Registry] Registered live face embedding successfully for user: {regno}")
    return True, "Face registered successfully!"


def recognize_face(image_data):
    """
    Decode a base64 JPEG, detect faces, and return the closest match.
    If the user has a live registered snapshot, perform a 1-to-1 cosine similarity verify.
    Returns a dict with: name, confidence, face_count
    """
    _load_model()

    np_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        return {'name': None, 'confidence': None, 'face_count': 0}

    # Keep image dimensions compact to prevent memory spikes
    h, w = img.shape[:2]
    if max(h, w) > 480:
        scale = 480.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    faces = _detector.get(img)
    face_count = len(faces) if faces else 0

    if not faces:
        return {'name': None, 'confidence': None, 'face_count': 0}

    regno = session.get('regno')

    # Analyze primary face (closest match)
    for face in faces:
        embedding = face.normed_embedding
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # 1-to-1 verification against dynamically registered face
        if regno and regno in _temp_embeddings:
            registered_emb = _temp_embeddings[regno]
            similarity = float(compute_cosine_similarity(embedding, registered_emb)[0][0])
            
            # Match threshold (InsightFace cosine similarity is highly precise)
            is_match = similarity >= 0.45
            
            return {
                'name': regno if is_match else "Mismatch",
                'confidence': f'{similarity:.2f}',
                'face_count': face_count
            }
        else:
            # Fallback to database general matching
            similarities = compute_cosine_similarity(embedding, _embeddings)
            idx = int(np.argmax(similarities))
            confidence = float(similarities[0][idx])
            return {
                'name': _names[idx],
                'confidence': f'{confidence:.2f}',
                'face_count': face_count
            }

    return {'name': None, 'confidence': None, 'face_count': face_count}


@recognition.route('/api/register_temp_face', methods=['POST'])
def register_temp_face_endpoint():
    """Endpoint called during Rules & Consent page to register live face embedding."""
    if 'regno' not in session:
        return jsonify({'success': False, 'message': 'Authentication required. Please log in again.'}), 401

    regno = session['regno']
    data = request.get_json(silent=True)
    if not data or 'image_data' not in data:
        return jsonify({'success': False, 'message': 'No webcam image data received.'}), 400

    image_data = data['image_data']
    try:
        success, message = register_temp_face(regno, image_data)
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        print(f"Error in register_temp_face: {e}")
        return jsonify({
            'success': False,
            'message': f"Server error: {str(e)}"
        }), 200


@recognition.route('/recognize', methods=['POST'])
def recognize():
    """REST endpoint for face recognition. Requires active session."""
    # Authentication check — only logged-in users can use recognition
    if 'regno' not in session:
        return jsonify({'error': 'Authentication required'}), 401

    image_data = request.json.get('image_data')
    if not image_data:
        return jsonify({'error': 'No image data provided'}), 400
    result = recognize_face(image_data)
    return jsonify(result)
