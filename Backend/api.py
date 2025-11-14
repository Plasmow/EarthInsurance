from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from embedding_match import get_alphaearth_record
    from risk_inference import predict_damage, predict_probability
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML modules not available: {e}")
    ML_AVAILABLE = False

app = Flask(__name__)
CORS(app)


def convert_to_expected_format(time_str):
    """Convert ISO 8601 timestamp to YYYY-MM-DD HH:MM:SS+HH:MM format."""
    if not time_str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00')
    
    try:
        if 'T' in time_str:
            if '.' in time_str:
                time_str = time_str.split('.')[0]
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(time_str.split('+')[0])
        
        return dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    
    except Exception as e:
        logger.error(f"Timestamp conversion error: {e}")
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00')


def get_risk_level(risk_score):
    """Map continuous risk score to human-readable label."""
    if risk_score >= 0.8:
        return 'Critique'
    elif risk_score >= 0.6:
        return 'Très élevé'
    elif risk_score >= 0.4:
        return 'Modéré'
    elif risk_score >= 0.2:
        return 'Faible'
    else:
        return 'Très faible'


def get_ef_scale_label(magnitude):
    """Convert EF-scale magnitude to descriptive label."""
    labels = {
        0: 'EF0 - Dégâts légers',
        1: 'EF1 - Dégâts modérés',
        2: 'EF2 - Dégâts considérables',
        3: 'EF3 - Dégâts sévères',
        4: 'EF4 - Dégâts dévastateurs',
        5: 'EF5 - Dégâts incroyables',
    }
    return labels.get(int(magnitude), 'Inconnu')


def calculate_risk_from_ml_models(latitude, longitude, time_utc=None):
    """Use ML models to predict tornado risk.
    
    Returns dict with risk metrics.
    """
    
    if not ML_AVAILABLE:
        logger.error("ML models not available")
        raise RuntimeError("ML inference engine not initialized")
    
    if time_utc is None:
        time_utc = datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00')
    else:
        time_utc = convert_to_expected_format(time_utc)
    
    try:
        logger.info(f"Generating embedding for lat={latitude}, lon={longitude}, time={time_utc}")
        embedding = get_alphaearth_record(latitude, longitude, time_utc)
        logger.info(f"Embedding generated: shape={len(embedding['embedding'])}")
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    try:
        logger.info("Calling predict_probability")
        probability, magnitude, magnitude_probs = predict_probability(
            embedding=embedding['embedding'],
            lat=latitude,
            lon=longitude,
            time_utc=time_utc,
            model_prob_dir="models_prob"
        )
        logger.info(f"Probability prediction: prob={probability}, mag={magnitude}")
        
    except Exception as e:
        logger.error(f"Error during probability prediction: {e}")
        probability, magnitude, magnitude_probs = 0.0, 0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        logger.info("Calling predict_damage")
        tornado_damage = predict_damage(
            embedding=embedding['embedding'],
            lat=latitude,
            lon=longitude,
            time_utc=time_utc,
            model_damage_dir="models_damage"
        )
        logger.info(f"Damage prediction: damage={tornado_damage}")
        
    except Exception as e:
        logger.error(f"Error during damage prediction: {e}")
        tornado_damage = 0.0
    
    # Risk score = fonction pondérée des magnitudes avec accent brutal sur les grosses
    # Différenciation drastique entre dangereux et pas dangereux
    
    damage_score_norm = float(tornado_damage / 10.0)  # 0-1 (normalize damage)
    
    # Fonction pondérée avec accent sur les grosses magnitudes
    if magnitude_probs and len(magnitude_probs) > 0:
        n_classes = len(magnitude_probs)
        # Créer les poids basés sur la taille réelle (2^0, 2^1, 2^2, ...)
        weights = np.array([2.0**i for i in range(n_classes)], dtype=float)
        weighted_probs = np.array(magnitude_probs, dtype=float) * weights
        weighted_probs = weighted_probs / (weighted_probs.sum() + 1e-8)  # renormaliser
        
        # Magnitude pondérée accentuée
        expected_magnitude_accentuated = sum(i * p for i, p in enumerate(weighted_probs))
        expected_mag_score = float(expected_magnitude_accentuated / (n_classes - 1)) if n_classes > 1 else 0.0
        
        # Calculer la probabilité d'EF1 ou plus (index >= 1)
        prob_ef1_plus = sum(magnitude_probs[i] for i in range(1, len(magnitude_probs)))
    else:
        mag_score = float(magnitude / 5.0)
        expected_mag_score = mag_score
        expected_magnitude_accentuated = magnitude
        prob_ef1_plus = 0.0
    
    # Fonction sigmoïde brutale pour différencier dangereux vs pas dangereux
    # Si mag_score < 0.3 → risque très faible (< 0.15)
    # Si mag_score >= 0.3 → risque monte rapidement
    if expected_mag_score < 0.15:
        # EF0 pur: TRÈS faible
        base_risk = 0.02
        risk_multiplier = 1.0
    elif expected_mag_score < 0.25:
        # Borderline EF0-EF1: faible
        base_risk = 0.08
        risk_multiplier = 1.1
    elif expected_mag_score < 0.35:
        # EF1 léger: modéré-bas
        base_risk = 0.25
        risk_multiplier = 1.5
    elif expected_mag_score < 0.5:
        # EF1-EF2 mix: modéré
        base_risk = 0.45
        risk_multiplier = 2.0
    elif expected_mag_score < 0.65:
        # EF2 dominant: élevé
        base_risk = 0.60
        risk_multiplier = 2.5
    elif expected_mag_score < 0.75:
        # EF3 dominant: très élevé
        base_risk = 0.75
        risk_multiplier = 3.0
    else:
        # EF4-5 dominant: CRITIQUE
        base_risk = 0.85
        risk_multiplier = 3.5
    
    risk_score = base_risk * risk_multiplier
    
    # BOOST BRUTAL si prob(EF1+) > 30%
    if prob_ef1_plus > 0.30:
        # Augmenter drastiquement
        risk_score = risk_score * 2.5  # Multiplier par 2.5
        logger.info(f"BOOST APPLIED: prob_ef1_plus={prob_ef1_plus:.2%} > 30%")
    
    # Clamp final à 0-1
    risk_score = float(np.clip(risk_score, 0, 1))
    
    logger.info(f"Final risk score: {risk_score} (mag_accentuated={expected_magnitude_accentuated:.2f}, mag_score={expected_mag_score:.2f}, prob_ef1+={prob_ef1_plus:.2%}, damage={tornado_damage:.2f})")
    
    return {
        'risk_score': risk_score,
        'probability': float(probability),
        'magnitude': int(magnitude),
        'magnitude_probs': [float(p) for p in magnitude_probs],
        'tornado_damage': float(tornado_damage),
        'ef_label': get_ef_scale_label(magnitude),
    }


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'EarthInsurance backend running',
        'ml_available': ML_AVAILABLE
    }), 200


@app.route('/api/calculate-risk', methods=['POST'])
def calculate_risk():
    """Compute tornado risk using ML models."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Empty request body'}), 400
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        time_utc = data.get('time_utc')
        
        if latitude is None or longitude is None:
            return jsonify({'error': 'latitude and longitude are required'}), 400
        
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return jsonify({'error': 'latitude and longitude must be numbers'}), 400
        
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({'error': 'Invalid coordinates (lat: -90 to 90, lon: -180 to 180)'}), 400
        
        ml_result = calculate_risk_from_ml_models(latitude, longitude, time_utc)
        
        return jsonify({
            'risk_score': ml_result['risk_score'],
            'risk_level': get_risk_level(ml_result['risk_score']),
            'probability': ml_result['probability'],
            'magnitude': ml_result['magnitude'],
            'magnitude_probs': ml_result['magnitude_probs'],
            'tornado_damage': ml_result['tornado_damage'],
            'ef_label': ml_result['ef_label'],
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00'),
            'model_info': 'ML-based tornado risk prediction using AlphaEarth embeddings'
        }), 200
    
    except Exception as e:
        logger.error(f"Error in /api/calculate-risk: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-calculate-risk', methods=['POST'])
def batch_calculate_risk():
    """Compute risks for multiple locations."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Empty request body'}), 400
        
        locations = data.get('locations', [])
        
        if not locations:
            return jsonify({'error': 'locations array is required'}), 400
        
        if not isinstance(locations, list):
            return jsonify({'error': 'locations must be an array'}), 400
        
        results = []
        
        for idx, loc in enumerate(locations):
            try:
                lat = loc.get('latitude')
                lng = loc.get('longitude')
                
                if lat is None or lng is None:
                    results.append({
                        'index': idx,
                        'error': 'latitude and longitude are required'
                    })
                    continue
                
                try:
                    lat = float(lat)
                    lng = float(lng)
                except (ValueError, TypeError):
                    results.append({
                        'index': idx,
                        'error': 'latitude and longitude must be numbers'
                    })
                    continue
                
                ml_result = calculate_risk_from_ml_models(lat, lng)
                
                results.append({
                    'index': idx,
                    'latitude': lat,
                    'longitude': lng,
                    'risk_score': ml_result['risk_score'],
                    'risk_level': get_risk_level(ml_result['risk_score']),
                    'probability': ml_result['probability'],
                    'magnitude': ml_result['magnitude'],
                    'magnitude_probs': ml_result['magnitude_probs'],
                    'tornado_damage': ml_result['tornado_damage'],
                    'ef_label': ml_result['ef_label'],
                })
            
            except Exception as e:
                logger.error(f"Error processing location {idx}: {e}")
                results.append({
                    'index': idx,
                    'error': str(e)
                })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        logger.error(f"Error in /api/batch-calculate-risk: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-zones', methods=['GET'])
def get_predefined_zones():
    """Return predefined risk zones with ML-derived risk metrics."""
    zones = [
        {
            'name': 'Tornado Alley (Oklahoma)',
            'latitude': 35.4676,
            'longitude': -97.5164,
            'risk_type': 'Tornadoes',
        },
        {
            'name': 'Kansas City Area',
            'latitude': 39.0997,
            'longitude': -94.5786,
            'risk_type': 'Tornadoes',
        },
        {
            'name': 'Dallas-Fort Worth',
            'latitude': 32.7767,
            'longitude': -96.7970,
            'risk_type': 'Tornadoes',
        },
    ]
    
    for zone in zones:
        try:
            ml_result = calculate_risk_from_ml_models(
                zone['latitude'],
                zone['longitude']
            )
            zone['risk_score'] = ml_result['risk_score']
            zone['risk_level'] = get_risk_level(ml_result['risk_score'])
            zone['probability'] = ml_result['probability']
            zone['magnitude'] = ml_result['magnitude']
            zone['magnitude_probs'] = ml_result['magnitude_probs']
            zone['tornado_damage'] = ml_result['tornado_damage']
            zone['ef_label'] = ml_result['ef_label']
        
        except Exception as e:
            logger.error(f"Error computing risk for {zone['name']}: {e}")
            zone['risk_score'] = 0.0
            zone['risk_level'] = 'Inconnu'
            zone['error'] = str(e)
    
    return jsonify({'zones': zones}), 200


@app.route('/api/predict-detailed', methods=['POST'])
def predict_detailed():
    """Detailed endpoint returning all ML model outputs."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Empty request body'}), 400
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        time_utc = data.get('time_utc')
        
        if latitude is None or longitude is None:
            return jsonify({'error': 'latitude and longitude are required'}), 400
        
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return jsonify({'error': 'latitude and longitude must be numbers'}), 400
        
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({'error': 'Invalid coordinates'}), 400
        
        ml_result = calculate_risk_from_ml_models(latitude, longitude, time_utc)
        
        return jsonify({
            'location': {
                'latitude': latitude,
                'longitude': longitude,
            },
            'timestamp': time_utc or datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00'),
            'probability_prediction': {
                'probability': ml_result['probability'],
                'magnitude': ml_result['magnitude'],
                'magnitude_probs': ml_result['magnitude_probs'],
                'ef_label': ml_result['ef_label'],
            },
            'damage_prediction': {
                'tornado_magnitude': ml_result['tornado_damage'],
            },
            'combined_risk': {
                'risk_score': ml_result['risk_score'],
                'risk_level': get_risk_level(ml_result['risk_score']),
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error in /api/predict-detailed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting EarthInsurance server…")
    print("📍 API available at http://localhost:5000")
    print("✅ CORS enabled")
    if ML_AVAILABLE:
        print("🤖 ML models loaded")
    else:
        print("⚠️  ML models NOT loaded (install dependencies)")
    print("📡 Waiting for requests…")
    app.run(debug=True, port=5000, host='0.0.0.0')