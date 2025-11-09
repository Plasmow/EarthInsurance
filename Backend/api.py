from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==============================================================================
# CALCUL DE RISQUE
# ==============================================================================

def calculate_risk_score(latitude, longitude):
    """
    Calcule le score de risque pour une localisation donnée.
    
    Args:
        latitude: Latitude de la localisation
        longitude: Longitude de la localisation
    
    Returns:
        float: Score de risque entre 0 et 1
    """
    
    # Zones à risque connus aux USA
    risk_zones = [
        # (lat, lng, risk_factor, radius_km)
        (37.7749, -122.4194, 0.9, 200),      # San Francisco - séismes
        (39.7392, -104.9903, 0.7, 300),      # Denver - tornades
        (35.1264, -97.0882, 0.85, 400),      # Oklahoma - tornades et séismes
        (29.7604, -95.3698, 0.75, 250),      # Houston - tempêtes, inondations
        (33.7490, -84.3880, 0.65, 280),      # Atlanta - tempêtes
    ]
    
    base_risk = 0.1  # Risque de base partout aux USA
    
    for zone_lat, zone_lng, zone_risk, radius in risk_zones:
        # Distance en degrés
        distance = np.sqrt((latitude - zone_lat)**2 + (longitude - zone_lng)**2)
        distance_km = distance * 111  # 1 degré ≈ 111 km
        
        if distance_km < radius:
            # Décroissance exponentielle du risque avec la distance
            proximity_factor = np.exp(-distance_km / (radius / 3))
            contribution = zone_risk * proximity_factor
            base_risk = max(base_risk, contribution)
    
    # Ajouter une composante aléatoire pour la variabilité
    noise = np.random.normal(0, 0.05)
    final_risk = np.clip(base_risk + noise, 0, 1)
    
    return float(final_risk)


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


def calculate_risk_with_ml(latitude, longitude):
    """
    Alternative: Utilise un modèle ML pré-entraîné pour prédire le risque.
    Tu pourrais charger ton modèle ici si tu en as un.
    """
    # Exemple avec features engineerées
    features = {
        'latitude': latitude,
        'longitude': longitude,
        'lat_squared': latitude ** 2,
        'lng_squared': longitude ** 2,
        'interaction': latitude * longitude,
    }
    
    # Simule une prédiction (remplace par ton vrai modèle)
    risk_score = 0.3 + 0.15 * np.sin(latitude / 50) + 0.15 * np.cos(longitude / 50)
    risk_score = np.clip(risk_score, 0, 1)
    
    return float(risk_score)


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
    Endpoint pour calculer le risque: HAZARD × EXPOSURE × VULNERABILITY
    
    Entrées (JSON POST):
    {
        "tornado_probability": 0.4,    # 0-1
        "ef_scale": 3,                  # 0-5
        "latitude": 35.4676,
        "longitude": -97.5164
    }
    """
    try:
        data = request.get_json()
        tornado_probability = data.get('tornado_probability')
        ef_scale = data.get('ef_scale')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        # Validation
        if any(x is None for x in [tornado_probability, ef_scale, latitude, longitude]):
            return jsonify({'error': 'Tous les paramètres requis'}), 400
        
        if not (0 <= tornado_probability <= 1):
            return jsonify({'error': 'tornado_probability entre 0 et 1'}), 400
        
        if not (0 <= ef_scale <= 5):
            return jsonify({'error': 'ef_scale entre 0 et 5'}), 400
        
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({'error': 'Coordonnées invalides'}), 400
        
        # Calcul avec le modèle HEV
        risk_data = calculate_risk_from_tornado_data(
            tornado_probability, 
            ef_scale, 
            latitude, 
            longitude
        )
        
        return jsonify({
            'risk_score': risk_data['risk_score'],
            'risk_level': get_risk_level(risk_data['risk_score']),
            'hazard': risk_data['hazard'],
            'exposure': risk_data['exposure'],
            'vulnerability': risk_data['vulnerability'],
            'tornado_probability': risk_data['tornado_probability'],
            'ef_scale': risk_data['ef_scale'],
            'ef_label': risk_data['ef_label'],
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': datetime.now().isoformat(),
            'formula': 'RISK = HAZARD × EXPOSURE × VULNERABILITY',
        }), 200
    
    except Exception as e:
        print(f"Erreur: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-calculate-risk', methods=['POST'])
def batch_calculate_risk():
    """
    Endpoint pour calculer les risques pour plusieurs localisations.
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
                risk_score = calculate_risk_score(lat, lng)
                results.append({
                    'latitude': lat,
                    'longitude': lng,
                    'risk_score': risk_score,
                    'risk_level': get_risk_level(risk_score),
                })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-zones', methods=['GET'])
def get_predefined_zones():
    """
    Retourne les zones de risque prédéfinies.
    """
    zones = [
        {
            'name': 'San Francisco Bay Area',
            'latitude': 37.7749,
            'longitude': -122.4194,
            'risk_type': 'Earthquakes',
            'base_risk': 0.9,
        },
        {
            'name': 'Tornado Alley',
            'latitude': 35.1264,
            'longitude': -97.0882,
            'risk_type': 'Tornadoes',
            'base_risk': 0.85,
        },
        {
            'name': 'Gulf Coast',
            'latitude': 29.7604,
            'longitude': -95.3698,
            'risk_type': 'Hurricanes & Storms',
            'base_risk': 0.75,
        },
    ]
    return jsonify({'zones': zones}), 200


# ==============================================================================
# LANCEMENT DU SERVEUR
# ==============================================================================

if __name__ == '__main__':
    print("🚀 Démarrage du serveur EarthInsurance...")
    print("📍 API disponible à http://localhost:5000")
    print("✅ CORS activé")
    print("📡 En attente de requêtes...")
    app.run(debug=True, port=5000, host='0.0.0.0')