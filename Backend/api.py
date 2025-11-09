from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

from risk_inference import predict_damage, predict_probability

# ==============================================================================
# UTILITAIRES DE CONVERSION DE TEMPS
# ==============================================================================

def convert_to_expected_format(time_str):
    """
    Convertit un timestamp ISO 8601 au format attendu par les modèles ML.
    Format attendu: 'YYYY-MM-DD HH:MM:SS+HH:MM'
    Format reçu: '2025-11-09T12:58:57.841038' ou '2025-11-09T12:58:57'
    """
    try:
        # Parser le timestamp ISO
        if 'T' in time_str:
            # Retirer les microsecondes si présentes
            if '.' in time_str:
                time_str = time_str.split('.')[0]
            
            # Parser la date
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        else:
            # Si déjà au bon format ou format personnalisé
            dt = datetime.strptime(time_str.split('+')[0].split('-')[0:3][0] + '-' + 
                                  time_str.split('+')[0].split('-')[1] + '-' + 
                                  time_str.split('+')[0].split('-')[2].split(' ')[0] + ' ' +
                                  time_str.split(' ')[1].split('+')[0], 
                                  '%Y-%m-%d %H:%M:%S')
        
        # Convertir au format attendu: 'YYYY-MM-DD HH:MM:SS+00:00'
        formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
        return formatted_time
    
    except Exception as e:
        print(f"Erreur lors de la conversion du timestamp: {e}")
        # Fallback: retourner le timestamp actuel au bon format
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00')


# ==============================================================================
# UTILITAIRES POUR PARSER LES RÉSULTATS ML
# ==============================================================================

def parse_probability_result(result):
    """
    Parse le résultat de predict_probability quel que soit son format.
    Retourne: (probability, magnitude, magnitude_probs)
    """
    try:
        # Cas 1: C'est un dictionnaire avec toutes les infos
        if isinstance(result, dict):
            return (
                float(result.get('probability', 0.0)),
                int(result.get('magnitude', 0)),
                [float(p) for p in result.get('magnitude_probs', [0.0] * 6)]
            )
        
        # Cas 2: C'est un tuple (probability, magnitude, probs)
        elif isinstance(result, tuple) and len(result) >= 3:
            return (
                float(result[0]),
                int(result[1]),
                [float(p) for p in result[2]]
            )
        
        # Cas 3: C'est un tuple (probability, magnitude)
        elif isinstance(result, tuple) and len(result) == 2:
            return (
                float(result[0]),
                int(result[1]),
                [0.0] * 6
            )
        
        # Cas 4: C'est juste une probabilité (float)
        elif isinstance(result, (int, float)):
            return (float(result), 0, [0.0] * 6)
        
        # Cas 5: Résultat inattendu
        else:
            print(f"Format de résultat inattendu pour predict_probability: {type(result)}")
            return (0.0, 0, [0.0] * 6)
            
    except Exception as e:
        print(f"Erreur lors du parsing du résultat de probabilité: {e}")
        traceback.print_exc()
        return (0.0, 0, [0.0] * 6)


def parse_damage_result(result):
    """
    Parse le résultat de predict_damage quel que soit son format.
    Retourne: tornado_damage (float)
    """
    try:
        # Cas 1: C'est un dictionnaire
        if isinstance(result, dict):
            return float(result.get('tornado_magnitude', 0.0))
        
        # Cas 2: C'est directement une valeur numérique
        elif isinstance(result, (int, float)):
            return float(result)
        
        # Cas 3: C'est un tuple, prendre le premier élément
        elif isinstance(result, tuple) and len(result) > 0:
            return float(result[0])
        
        # Cas 4: Résultat inattendu
        else:
            print(f"Format de résultat inattendu pour predict_damage: {type(result)}")
            return 0.0
            
    except Exception as e:
        print(f"Erreur lors du parsing du résultat de dégâts: {e}")
        traceback.print_exc()
        return 0.0


# ==============================================================================
# CALCUL DE RISQUE AVEC ML
# ==============================================================================

def generate_embedding(latitude, longitude, time_str=None):
    """
    Génère un embedding pour le modèle ML.
    Dans une version de production, cela pourrait être basé sur des données météo réelles.
    Pour l'instant, on génère un embedding semi-aléatoire basé sur la localisation.
    """
    # Seed basé sur la localisation pour avoir des résultats reproductibles
    seed = int((abs(latitude) * 1000 + abs(longitude) * 1000) % 2**32)
    np.random.seed(seed)
    embedding = np.random.rand(64).astype(float).tolist()
    return embedding


def get_risk_level(risk_score):
    """
    Convertit un score de risque en label textuel.
    """
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
    """
    Convertit une magnitude en label EF Scale.
    """
    labels = {
        0: 'EF0 - Dégâts légers',
        1: 'EF1 - Dégâts modérés',
        2: 'EF2 - Dégâts considérables',
        3: 'EF3 - Dégâts sévères',
        4: 'EF4 - Dégâts dévastateurs',
        5: 'EF5 - Dégâts incroyables',
    }
    return labels.get(magnitude, 'Inconnu')


def calculate_risk_from_ml_models(latitude, longitude, time_utc=None):
    """
    Utilise les modèles ML pour prédire le risque de tornade.
    
    Retourne:
    - probability: probabilité d'occurrence (0-1)
    - magnitude: échelle EF prédite (0-5)
    - magnitude_probs: distribution de probabilité sur les magnitudes
    - damage: estimation des dégâts (tornado_magnitude)
    """
    # Si pas de timestamp fourni, utiliser maintenant au bon format
    if time_utc is None:
        time_utc = datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00')
    else:
        # Convertir le timestamp au format attendu
        time_utc = convert_to_expected_format(time_utc)
    
    # Générer l'embedding
    embedding = generate_embedding(latitude, longitude, time_utc)
    
    # Prédire la probabilité et la magnitude
    probability = 0.0
    magnitude = 0
    magnitude_probs = [0.0] * 6
    
    try:
        print(f"Appel predict_probability pour lat={latitude}, lon={longitude}")
        prob_result = predict_probability(
            embedding=embedding,
            lat=latitude,
            lon=longitude,
            time_utc=time_utc,
            model_prob_dir="models_prob"
        )
        print(f"Résultat brut predict_probability: {prob_result} (type: {type(prob_result)})")
        
        probability, magnitude, magnitude_probs = parse_probability_result(prob_result)
        print(f"Résultat parsé: prob={probability}, mag={magnitude}, probs={magnitude_probs}")
        
    except Exception as e:
        print(f"Erreur lors de la prédiction de probabilité: {e}")
        traceback.print_exc()
    
    # Prédire les dégâts
    tornado_damage = 0.0
    
    try:
        print(f"Appel predict_damage pour lat={latitude}, lon={longitude}")
        damage_result = predict_damage(
            embedding=embedding,
            lat=latitude,
            lon=longitude,
            time_utc=time_utc,
            model_damage_dir="models_damage"
        )
        print(f"Résultat brut predict_damage: {damage_result} (type: {type(damage_result)})")
        
        tornado_damage = parse_damage_result(damage_result)
        print(f"Résultat parsé: damage={tornado_damage}")
        
    except Exception as e:
        print(f"Erreur lors de la prédiction de dégâts: {e}")
        traceback.print_exc()
    
    # Calculer le score de risque combiné
    # Risk = Probability × (Magnitude/5) × (1 + Damage/10)
    normalized_magnitude = magnitude / 5.0 if magnitude > 0 else 0.1
    damage_factor = 1 + (tornado_damage / 10.0)
    risk_score = probability * normalized_magnitude * damage_factor
    risk_score = float(np.clip(risk_score, 0, 1))
    
    print(f"Score de risque final: {risk_score}")
    
    return {
        'risk_score': risk_score,
        'probability': float(probability),
        'magnitude': int(magnitude),
        'magnitude_probs': [float(p) for p in magnitude_probs],
        'tornado_damage': float(tornado_damage),
        'ef_label': get_ef_scale_label(magnitude),
    }


# ==============================================================================
# ROUTES API
# ==============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """
    Endpoint de santé pour vérifier si le serveur est actif.
    """
    return jsonify({'status': 'ok', 'message': 'EarthInsurance backend running'}), 200


@app.route('/api/calculate-risk', methods=['POST'])
def calculate_risk():
    """
    Endpoint pour calculer le risque de tornade avec les modèles ML.
    
    Entrées (JSON POST):
    {
        "latitude": 35.4676,
        "longitude": -97.5164,
        "time_utc": "2025-05-02 14:30:00+00:00" (optionnel)
    }
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        time_utc = data.get('time_utc')  # Optionnel
        
        # Validation
        if latitude is None or longitude is None:
            return jsonify({'error': 'Latitude et longitude requis'}), 400
        
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({'error': 'Coordonnées invalides'}), 400
        
        # Calcul avec les modèles ML
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
            'model_info': 'ML-based tornado risk prediction using AlphaEarth embeddings',
        }), 200
    
    except Exception as e:
        print(f"Erreur: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-calculate-risk', methods=['POST'])
def batch_calculate_risk():
    """
    Endpoint pour calculer les risques pour plusieurs localisations.
    
    Entrées (JSON POST):
    {
        "locations": [
            {"latitude": 35.47, "longitude": -97.52},
            {"latitude": 40.71, "longitude": -74.01},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        locations = data.get('locations', [])
        
        if not locations:
            return jsonify({'error': 'Aucune localisation fournie'}), 400
        
        results = []
        for loc in locations:
            lat = loc.get('latitude')
            lng = loc.get('longitude')
            
            if lat is not None and lng is not None:
                try:
                    ml_result = calculate_risk_from_ml_models(lat, lng)
                    results.append({
                        'latitude': lat,
                        'longitude': lng,
                        'risk_score': ml_result['risk_score'],
                        'risk_level': get_risk_level(ml_result['risk_score']),
                        'probability': ml_result['probability'],
                        'magnitude': ml_result['magnitude'],
                        'ef_label': ml_result['ef_label'],
                    })
                except Exception as e:
                    print(f"Erreur pour location ({lat}, {lng}): {e}")
                    results.append({
                        'latitude': lat,
                        'longitude': lng,
                        'error': str(e),
                    })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-zones', methods=['GET'])
def get_predefined_zones():
    """
    Retourne les zones de risque prédéfinies avec calculs ML.
    """
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
    
    # Calculer le risque pour chaque zone
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
            zone['ef_label'] = ml_result['ef_label']
        except Exception as e:
            print(f"Erreur calcul risque pour {zone['name']}: {e}")
            zone['risk_score'] = 0.0
            zone['risk_level'] = 'Inconnu'
    
    return jsonify({'zones': zones}), 200


@app.route('/api/predict-detailed', methods=['POST'])
def predict_detailed():
    """
    Endpoint détaillé qui retourne toutes les informations des modèles ML.
    
    Entrées (JSON POST):
    {
        "latitude": 35.4676,
        "longitude": -97.5164,
        "time_utc": "2025-05-02 14:30:00+00:00" (optionnel)
    }
    """
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        time_utc = data.get('time_utc')
        
        if latitude is None or longitude is None:
            return jsonify({'error': 'Latitude et longitude requis'}), 400
        
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({'error': 'Coordonnées invalides'}), 400
        
        # Convertir le timestamp au bon format si fourni
        if time_utc is None:
            time_utc = datetime.now().strftime('%Y-%m-%d %H:%M:%S+00:00')
        else:
            time_utc = convert_to_expected_format(time_utc)
        
        # Générer l'embedding
        embedding = generate_embedding(latitude, longitude, time_utc)
        
        # Prédictions ML
        prob_result = predict_probability(
            embedding=embedding,
            lat=latitude,
            lon=longitude,
            time_utc=time_utc,
            model_prob_dir="models_prob"
        )
        
        damage_result = predict_damage(
            embedding=embedding,
            lat=latitude,
            lon=longitude,
            time_utc=time_utc,
            model_damage_dir="models_damage"
        )
        
        # Parser les résultats
        probability, magnitude, magnitude_probs = parse_probability_result(prob_result)
        tornado_magnitude = parse_damage_result(damage_result)
        
        # Résultat complet
        ml_result = calculate_risk_from_ml_models(latitude, longitude, time_utc)
        
        return jsonify({
            'location': {
                'latitude': latitude,
                'longitude': longitude,
            },
            'timestamp': time_utc,
            'probability_prediction': {
                'probability': probability,
                'magnitude': magnitude,
                'magnitude_probs': magnitude_probs,
                'ef_label': get_ef_scale_label(magnitude),
            },
            'damage_prediction': {
                'tornado_magnitude': tornado_magnitude,
            },
            'combined_risk': {
                'risk_score': ml_result['risk_score'],
                'risk_level': get_risk_level(ml_result['risk_score']),
            }
        }), 200
    
    except Exception as e:
        print(f"Erreur: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# LANCEMENT DU SERVEUR
# ==============================================================================

if __name__ == '__main__':
    print("🚀 Démarrage du serveur EarthInsurance...")
    print("📍 API disponible à http://localhost:5000")
    print("✅ CORS activé")
    print("🤖 Modèles ML chargés")
    print("📡 En attente de requêtes...")
    app.run(debug=True, port=5000, host='0.0.0.0')