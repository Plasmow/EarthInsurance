import React, { useState, useRef, useEffect } from 'react';
import { MapContainer, TileLayer, useMapEvents, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix pour les icones Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Fonction pour obtenir la couleur selon le risque
const getRiskColor = (riskScore) => {
  if (riskScore >= 0.8) return '#8B0000'; // Très élevé - rouge foncé
  if (riskScore >= 0.6) return '#FF4500'; // Élevé - rouge-orange
  if (riskScore >= 0.4) return '#FFD700'; // Modéré - jaune-or
  if (riskScore >= 0.2) return '#90EE90'; // Faible - vert clair
  return '#00AA00'; // Très faible - vert
};

// Fonction pour obtenir le texte du risque
const getRiskLevel = (riskScore) => {
  if (riskScore >= 0.8) return 'Critique';
  if (riskScore >= 0.6) return 'Très élevé';
  if (riskScore >= 0.4) return 'Modéré';
  if (riskScore >= 0.2) return 'Faible';
  return 'Très faible';
};

// Composant pour gérer les clics sur la carte
function MapClickHandler({ onLocationClick }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      onLocationClick(lat, lng);
    },
  });
  return null;
}

// Composant pour afficher les zones de risque
function RiskCircles({ riskZones }) {
  const map = useMap();

  useEffect(() => {
    const circles = [];
    
    riskZones.forEach((zone) => {
      const circle = L.circleMarker(
        [zone.lat, zone.lng],
        {
          radius: 30,
          fillColor: zone.color,
          color: zone.color,
          weight: 2,
          opacity: 0.8,
          fillOpacity: 0.6,
        }
      ).bindPopup(
        `<div class="p-3">
          <p class="font-bold text-lg">${zone.riskLevel}</p>
          <p class="text-sm">Score: ${(zone.riskScore * 100).toFixed(1)}%</p>
          <p class="text-xs text-gray-600">Lat: ${zone.lat.toFixed(4)}</p>
          <p class="text-xs text-gray-600">Lng: ${zone.lng.toFixed(4)}</p>
        </div>`
      ).addTo(map);
      circles.push(circle);
    });

    return () => {
      circles.forEach(circle => map.removeLayer(circle));
    };
  }, [riskZones, map]);

  return null;
}

export default function App() {
  const [riskZones, setRiskZones] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const mapRef = useRef(null);
  const [mapCenter] = useState([39.8283, -98.5795]); // Centre des USA
  const [zoom] = useState(4);

  const handleLocationClick = async (lat, lng) => {
    setLoading(true);
    setError(null);

    try {
      // Appel au backend pour calculer le risque
      const response = await fetch('http://localhost:5000/api/calculate-risk', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          latitude: lat,
          longitude: lng,
        }),
      });

      if (!response.ok) {
        throw new Error('Erreur lors du calcul du risque');
      }

      const data = await response.json();
      const riskScore = data.risk_score || 0;

      // Ajouter la nouvelle zone de risque
      const newZone = {
        id: Date.now(),
        lat,
        lng,
        riskScore,
        color: getRiskColor(riskScore),
        riskLevel: getRiskLevel(riskScore),
      };

      setRiskZones([...riskZones, newZone]);
    } catch (err) {
      setError(err.message);
      console.error('Erreur:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClearZones = () => {
    setRiskZones([]);
    setError(null);
  };

  return (
    <div className="w-full h-screen flex flex-col bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-4 shadow-lg">
        <h1 className="text-3xl font-bold">🌍 EarthInsurance Risk Map</h1>
        <p className="text-blue-100 mt-1">Cliquez sur la carte pour analyser le risque de catastrophes naturelles</p>
      </div>

      {/* Main content */}
      <div className="flex-1 flex gap-4 p-4">
        {/* Map */}
        <div className="flex-1 rounded-lg overflow-hidden shadow-2xl border-2 border-blue-400">
          <MapContainer
            center={mapCenter}
            zoom={zoom}
            style={{ width: '100%', height: '100%' }}
            ref={mapRef}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; OpenStreetMap contributors'
              maxZoom={18}
            />
            <MapClickHandler onLocationClick={handleLocationClick} />
            <RiskCircles riskZones={riskZones} />
          </MapContainer>
        </div>

        {/* Sidebar */}
        <div className="w-80 bg-slate-800 rounded-lg shadow-2xl p-6 flex flex-col border border-blue-500">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span className="text-2xl">📊</span> Analyses
          </h2>

          {/* Status */}
          <div className="mb-4">
            {loading && (
              <div className="bg-blue-900 text-blue-100 p-3 rounded-lg flex items-center gap-2">
                <div className="animate-spin">⏳</div>
                <span>Calcul en cours...</span>
              </div>
            )}
            {error && (
              <div className="bg-red-900 text-red-100 p-3 rounded-lg">
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Risk zones list */}
          <div className="flex-1 overflow-y-auto mb-4">
            {riskZones.length === 0 ? (
              <div className="text-center text-slate-400 py-8">
                <p className="text-4xl mb-2">📍</p>
                <p>Aucune analyse pour le moment</p>
                <p className="text-xs text-slate-500 mt-2">Cliquez sur la carte pour commencer</p>
              </div>
            ) : (
              <div className="space-y-3">
                {riskZones.map((zone) => (
                  <div
                    key={zone.id}
                    className="bg-slate-700 rounded-lg p-4 border-l-4 hover:bg-slate-600 transition"
                    style={{ borderLeftColor: zone.color }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span
                        className="px-3 py-1 rounded-full text-white text-sm font-bold"
                        style={{ backgroundColor: zone.color }}
                      >
                        {zone.riskLevel}
                      </span>
                      <span className="text-xs text-slate-400">
                        {(zone.riskScore * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="text-xs text-slate-300 space-y-1">
                      <p>📍 Lat: {zone.lat.toFixed(4)}</p>
                      <p>📍 Lng: {zone.lng.toFixed(4)}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Stats */}
          {riskZones.length > 0 && (
            <div className="bg-slate-700 rounded-lg p-4 mb-4">
              <p className="text-sm text-slate-300 mb-2">
                <span className="font-bold text-white">{riskZones.length}</span> localisation(s) analysée(s)
              </p>
              <div className="text-xs text-slate-400 space-y-1">
                <p>🔴 Risque moyen: {((riskZones.reduce((sum, z) => sum + z.riskScore, 0) / riskZones.length) * 100).toFixed(1)}%</p>
              </div>
            </div>
          )}

          {/* Actions */}
          <button
            onClick={handleClearZones}
            disabled={riskZones.length === 0}
            className="w-full bg-red-600 hover:bg-red-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white py-2 rounded-lg font-semibold transition"
          >
            🗑️ Effacer tout
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-slate-900 border-t border-blue-500 px-4 py-3 text-center text-slate-400 text-sm">
        💡 Indicateur: Vert (faible) → Jaune (modéré) → Orange (élevé) → Rouge (critique)
      </div>
    </div>
  );
}