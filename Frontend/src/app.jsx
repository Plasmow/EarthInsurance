import React, { useState, useRef, useEffect } from 'react';
import { MapContainer, TileLayer, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const getRiskColor = (riskScore) => {
  if (riskScore >= 0.8) return '#8B0000';
  if (riskScore >= 0.6) return '#FF4500';
  if (riskScore >= 0.4) return '#FFD700';
  if (riskScore >= 0.2) return '#90EE90';
  return '#00AA00';
};

const getRiskLevel = (riskScore) => {
  if (riskScore >= 0.8) return 'Critique';
  if (riskScore >= 0.6) return 'Très élevé';
  if (riskScore >= 0.4) return 'Modéré';
  if (riskScore >= 0.2) return 'Faible';
  return 'Très Faible';
};

function MapClickHandler({ onLocationClick }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      onLocationClick(lat, lng);
    },
  });
  return null;
}

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
          weight: 3,
          opacity: 0.9,
          fillOpacity: 0.5,
        }
      ).bindPopup(`
        <div style="min-width: 280px; font-family: system-ui;">
          <div style="background: ${zone.color}; color: white; padding: 12px; margin: -10px -10px 10px -10px; border-radius: 4px 4px 0 0;">
            <h3 style="margin: 0; font-size: 18px; font-weight: bold;">${zone.riskLevel}</h3>
            <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 13px;">Score: ${(zone.riskScore * 100).toFixed(1)}%</p>
          </div>
          
          <div style="padding: 8px 0;">
            <div style="margin-bottom: 12px;">
              <p style="margin: 0 0 6px 0; font-weight: 600; font-size: 13px; color: #333;">📍 Localisation</p>
              <p style="margin: 0; font-size: 12px; color: #666;">Lat: ${zone.lat.toFixed(4)}° | Lng: ${zone.lng.toFixed(4)}°</p>
            </div>

            <div style="margin-bottom: 12px;">
              <p style="margin: 0 0 6px 0; font-weight: 600; font-size: 13px; color: #333;">⚠️  Magnitude</p>
              <p style="margin: 0; font-size: 12px; color: #666;">Magnitude: ${zone.ef_label}</p>
            </div>

            <div style="margin-bottom: 8px;">
              <p style="margin: 0 0 6px 0; font-weight: 600; font-size: 13px; color: #333;">💥 Dégâts estimés</p>
              <p style="margin: 0; font-size: 12px; color: #666;">Niveau: ${zone.tornado_damage.toFixed(2)}/10</p>
            </div>

            ${zone.magnitude_probs && zone.magnitude_probs.length > 0 ? `
              <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e5e5;">
                <p style="margin: 0 0 6px 0; font-weight: 600; font-size: 13px; color: #333;">📊 Distribution EF Scale</p>
                ${zone.magnitude_probs.map((prob, idx) => `
                  <div style="display: flex; align-items: center; margin: 3px 0;">
                    <span style="font-size: 11px; color: #666; width: 35px;">EF${idx}:</span>
                    <div style="flex: 1; background: #e5e5e5; height: 14px; border-radius: 7px; overflow: hidden;">
                      <div style="background: ${zone.color}; height: 100%; width: ${(prob * 100).toFixed(0)}%; transition: width 0.3s;"></div>
                    </div>
                    <span style="font-size: 11px; color: #666; margin-left: 6px; width: 40px; text-align: right;">${(prob * 100).toFixed(1)}%</span>
                  </div>
                `).join('')}
              </div>
            ` : ''}
          </div>
        </div>
      `).addTo(map);
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
  const [apiStatus, setApiStatus] = useState('unknown');
  const mapRef = useRef(null);
  const [mapCenter] = useState([39.8283, -98.5795]);
  const [zoom] = useState(4);

  useEffect(() => {
    const checkAPI = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/health', {
          headers: { 'Content-Type': 'application/json' },
        });
        if (response.ok) {
          const data = await response.json();
          setApiStatus(data.ml_available ? 'ready' : 'no-models');
        } else {
          setApiStatus('error');
        }
      } catch (err) {
        setApiStatus('offline');
      }
    };

    checkAPI();
  }, []);

  const handleLocationClick = async (lat, lng) => {
    setLoading(true);
    setError(null);

    try {
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
        const errorData = await response.json();
        throw new Error(errorData.error || `API returned ${response.status}`);
      }

      const data = await response.json();
      setApiStatus('ready');

      const newZone = {
        id: Date.now(),
        lat,
        lng,
        riskScore: data.risk_score || 0,
        probability: data.probability || 0,
        magnitude: data.magnitude || 0,
        magnitude_probs: data.magnitude_probs || [],
        tornado_damage: data.tornado_damage || 0,
        ef_label: data.ef_label || 'EF0',
        color: getRiskColor(data.risk_score || 0),
        riskLevel: data.risk_level || 'Inconnu',
        timestamp: new Date().toLocaleString(),
      };

      setRiskZones([...riskZones, newZone]);
    } catch (err) {
      setError(err.message || 'Erreur lors du calcul du risque');
      setApiStatus('offline');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClearZones = () => {
    setRiskZones([]);
    setError(null);
  };

  return (
    <div className="app-container">
      <style>{`
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
          overflow: hidden;
        }

        .app-container {
          width: 100vw;
          height: 100vh;
          display: flex;
          flex-direction: column;
          background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }

        .header {
          background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
          color: white;
          padding: 20px 32px;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
          position: relative;
          overflow: hidden;
        }

        .header::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
          animation: shimmer 3s infinite;
        }

        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }

        .header-content {
          position: relative;
          z-index: 1;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .header-left h1 {
          font-size: 32px;
          font-weight: 800;
          margin-bottom: 8px;
          text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .header-left p {
          color: rgba(255,255,255,0.9);
          font-size: 15px;
        }

        .api-status-badge {
          display: flex;
          align-items: center;
          gap: 8px;
          background: rgba(255,255,255,0.2);
          padding: 8px 14px;
          border-radius: 20px;
          font-size: 13px;
          font-weight: 600;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        .status-dot.ready {
          background: #10b981;
        }

        .status-dot.no-models {
          background: #f59e0b;
        }

        .status-dot.offline {
          background: #ef4444;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .main-content {
          flex: 1;
          display: flex;
          gap: 20px;
          padding: 20px;
          overflow: hidden;
        }

        .map-wrapper {
          flex: 1;
          border-radius: 16px;
          overflow: hidden;
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4);
          border: 2px solid rgba(59, 130, 246, 0.5);
          position: relative;
        }

        .map-wrapper::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          border-radius: 16px;
          padding: 2px;
          background: linear-gradient(45deg, #3b82f6, #8b5cf6);
          -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          -webkit-mask-composite: xor;
          mask-composite: exclude;
          pointer-events: none;
          z-index: 1000;
        }

        .sidebar {
          width: 380px;
          background: rgba(30, 41, 59, 0.95);
          backdrop-filter: blur(10px);
          border-radius: 16px;
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
          padding: 24px;
          display: flex;
          flex-direction: column;
          border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .sidebar-header {
          color: white;
          font-size: 22px;
          font-weight: 700;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .sidebar-header span {
          font-size: 28px;
        }

        .status-container {
          margin-bottom: 20px;
        }

        .status-loading, .status-error {
          padding: 14px 16px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 14px;
          font-weight: 500;
          animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .status-loading {
          background: rgba(30, 58, 138, 0.8);
          color: #bfdbfe;
          border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .status-error {
          background: rgba(127, 29, 29, 0.8);
          color: #fecaca;
          border: 1px solid rgba(220, 38, 38, 0.3);
        }

        .spinner {
          display: inline-block;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .zones-list {
          flex: 1;
          overflow-y: auto;
          margin-bottom: 20px;
          padding-right: 8px;
        }

        .zones-list::-webkit-scrollbar {
          width: 6px;
        }

        .zones-list::-webkit-scrollbar-track {
          background: rgba(0,0,0,0.2);
          border-radius: 3px;
        }

        .zones-list::-webkit-scrollbar-thumb {
          background: rgba(59, 130, 246, 0.5);
          border-radius: 3px;
        }

        .zones-list::-webkit-scrollbar-thumb:hover {
          background: rgba(59, 130, 246, 0.7);
        }

        .empty-state {
          text-align: center;
          color: #94a3b8;
          padding: 60px 20px;
        }

        .empty-state-icon {
          font-size: 64px;
          margin-bottom: 16px;
          opacity: 0.5;
        }

        .empty-state p {
          font-size: 16px;
          margin-bottom: 8px;
        }

        .empty-state-hint {
          font-size: 13px;
          color: #64748b;
          margin-top: 12px;
        }

        .zone-card {
          background: rgba(51, 65, 85, 0.8);
          border-radius: 12px;
          padding: 18px;
          margin-bottom: 14px;
          border-left: 4px solid;
          transition: all 0.3s ease;
          cursor: pointer;
          animation: fadeIn 0.4s ease;
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        .zone-card:hover {
          background: rgba(71, 85, 105, 0.8);
          transform: translateX(4px);
          box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }

        .zone-card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 14px;
        }

        .risk-badge {
          padding: 6px 14px;
          border-radius: 20px;
          color: white;
          font-size: 13px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .zone-score {
          font-size: 12px;
          color: #cbd5e1;
          font-weight: 600;
        }

        .zone-details {
          color: #cbd5e1;
          font-size: 12px;
          line-height: 1.6;
        }

        .zone-detail-row {
          display: flex;
          justify-content: space-between;
          padding: 4px 0;
        }

        .zone-detail-label {
          color: #94a3b8;
          font-weight: 500;
        }

        .zone-detail-value {
          color: #e2e8f0;
          font-weight: 600;
        }

        .zone-probs {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid rgba(148, 163, 184, 0.2);
        }

        .zone-probs-title {
          font-size: 11px;
          color: #94a3b8;
          margin-bottom: 8px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .prob-bar {
          display: flex;
          align-items: center;
          margin: 4px 0;
        }

        .prob-label {
          font-size: 11px;
          color: #94a3b8;
          width: 40px;
        }

        .prob-track {
          flex: 1;
          background: rgba(0,0,0,0.3);
          height: 12px;
          border-radius: 6px;
          overflow: hidden;
          margin: 0 8px;
        }

        .prob-fill {
          height: 100%;
          transition: width 0.5s ease;
          border-radius: 6px;
        }

        .prob-value {
          font-size: 11px;
          color: #cbd5e1;
          width: 45px;
          text-align: right;
          font-weight: 600;
        }

        .stats-card {
          background: rgba(51, 65, 85, 0.8);
          border-radius: 12px;
          padding: 16px;
          margin-bottom: 16px;
          border: 1px solid rgba(59, 130, 246, 0.2);
        }

        .stats-card p {
          color: #cbd5e1;
          font-size: 14px;
          margin-bottom: 12px;
        }

        .stats-number {
          font-weight: 700;
          color: white;
          font-size: 18px;
        }

        .stats-details {
          font-size: 12px;
          color: #94a3b8;
          line-height: 1.8;
        }

        .clear-button {
          width: 100%;
          background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
          color: white;
          padding: 14px;
          border-radius: 10px;
          border: none;
          font-weight: 700;
          font-size: 15px;
          cursor: pointer;
          transition: all 0.3s ease;
          box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }

        .clear-button:hover:not(:disabled) {
          background: linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%);
          transform: translateY(-2px);
          box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        }

        .clear-button:disabled {
          background: #475569;
          cursor: not-allowed;
          opacity: 0.5;
        }

        .footer {
          background: #0f172a;
          border-top: 1px solid rgba(59, 130, 246, 0.3);
          padding: 14px 24px;
          text-align: center;
          color: #94a3b8;
          font-size: 13px;
        }

        .leaflet-popup-content-wrapper {
          border-radius: 8px;
          box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }

        .leaflet-popup-content {
          margin: 10px;
        }
      `}</style>

      <div className="header">
        <div className="header-content">
          <div className="header-left">
            <h1>🌍 EarthInsurance Risk Map</h1>
            <p>Cliquez sur la carte pour analyser le risque de tornade avec ML</p>
          </div>
          <div className="api-status-badge">
            <div className={`status-dot ${apiStatus}`}></div>
            <span>
              {apiStatus === 'ready' && 'API Prête'}
              {apiStatus === 'no-models' && 'API (pas de modèles)'}
              {apiStatus === 'offline' && 'Offline'}
              {apiStatus === 'error' && 'Erreur API'}
              {apiStatus === 'unknown' && 'Vérification...'}
            </span>
          </div>
        </div>
      </div>

      <div className="main-content">
        <div className="map-wrapper">
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

        <div className="sidebar">
          <div className="sidebar-header">
            <span>📊</span>
            <span>Analyses ML</span>
          </div>

          <div className="status-container">
            {loading && (
              <div className="status-loading">
                <span className="spinner">⏳</span>
                <span>Calcul ML en cours...</span>
              </div>
            )}
            {error && (
              <div className="status-error">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="zones-list">
            {riskZones.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">📍</div>
                <p>Aucune analyse pour le moment</p>
                <p className="empty-state-hint">Cliquez sur la carte pour commencer</p>
              </div>
            ) : (
              riskZones.map((zone) => (
                <div
                  key={zone.id}
                  className="zone-card"
                  style={{ borderLeftColor: zone.color }}
                >
                  <div className="zone-card-header">
                    <span
                      className="risk-badge"
                      style={{ backgroundColor: zone.color }}
                    >
                      {zone.riskLevel}
                    </span>
                    <span className="zone-score">
                      {(zone.riskScore * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="zone-details">
                    <div className="zone-detail-row">
                      <span className="zone-detail-label">📍 Position</span>
                      <span className="zone-detail-value">
                        {zone.lat.toFixed(4)}°, {zone.lng.toFixed(4)}°
                      </span>
                    </div>
                    <div className="zone-detail-row">
                      <span className="zone-detail-label">🌪️ Magnitude</span>
                      <span className="zone-detail-value">{zone.ef_label}</span>
                    </div>
                    <div className="zone-detail-row">
                      <span className="zone-detail-label">💥 Dégâts</span>
                      <span className="zone-detail-value">
                        {zone.tornado_damage.toFixed(2)}/10
                      </span>
                    </div>
                  </div>

                  {zone.magnitude_probs && zone.magnitude_probs.length > 0 && (
                    <div className="zone-probs">
                      <div className="zone-probs-title">Distribution EF Scale</div>
                      {zone.magnitude_probs.map((prob, idx) => (
                        <div key={idx} className="prob-bar">
                          <span className="prob-label">EF{idx}</span>
                          <div className="prob-track">
                            <div
                              className="prob-fill"
                              style={{
                                width: `${(prob * 100).toFixed(0)}%`,
                                backgroundColor: zone.color,
                              }}
                            />
                          </div>
                          <span className="prob-value">
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          {riskZones.length > 0 && (
            <div className="stats-card">
              <p>
                <span className="stats-number">{riskZones.length}</span> localisation(s) analysée(s)
              </p>
              <div className="stats-details">
                🔴 Risque moyen:{' '}
                {((riskZones.reduce((sum, z) => sum + z.riskScore, 0) / riskZones.length) * 100).toFixed(1)}%
              </div>
            </div>
          )}

          <button
            onClick={handleClearZones}
            disabled={riskZones.length === 0}
            className="clear-button"
          >
            🗑️ Effacer tout
          </button>
        </div>
      </div>

      <div className="footer">
        💡 Indicateur: Vert (faible) → Jaune (modéré) → Orange (élevé) → Rouge (critique)
      </div>
    </div>
  );
}
