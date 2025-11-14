import ee
import numpy as np
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

project = "gen-lang-client-0546266030"

def get_alphaearth_record(lat: float,
                          lon: float,
                          when: str,
                          buffer_m: int = 500,
                          scale: int = 30):
    """
    Renvoie un dict EXACTEMENT du format:
    {
      "embedding": [float]*64,
      "lat": <lat>,
      "lon": <lon>,
      "time_utc": "YYYY-MM-DD HH:MM:SS+00:00",
    }
    
    Gère les cas où AlphaEarth n'a pas de données (années futures, zones non couvertes)
    """
    try:
        # Init EE
        ee.Initialize(project=project)

        # Fenêtre annuelle basée sur `when`
        dt = datetime.fromisoformat(when).date()
        year = dt.year
        start, end = f"{year}-01-01", f"{year+1}-01-01"

        # Point (lon, lat) en WGS84
        pt = ee.Geometry.Point([float(lon), float(lat)])

        # Tuile AlphaEarth qui couvre le point sur l'année
        col = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
               .filterDate(start, end)
               .filterBounds(pt))
        
        size = col.size().getInfo()
        
        if size == 0:
            logger.warning(f"⚠️ Aucune tuile AlphaEarth pour {year} à ({lat}, {lon})")
            logger.warning(f"   → Tentative avec l'année précédente...")
            
            # FALLBACK: Essayer l'année précédente
            year_prev = year - 1
            start_prev = f"{year_prev}-01-01"
            end_prev = f"{year}-01-01"
            
            col = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
                   .filterDate(start_prev, end_prev)
                   .filterBounds(pt))
            
            size = col.size().getInfo()
            
            if size == 0:
                logger.warning(f"   → Pas de données pour {year_prev} non plus")
                logger.warning(f"   → Utilisation d'embeddings simulés")
                return _generate_fallback_embedding(lat, lon, when)

        img = col.first()

        # Pixel au point ; sinon on cherche dans un léger buffer
        feat = img.sample(region=pt, scale=scale, numPixels=1, geometries=False).first()
        
        if (feat is None) and buffer_m and buffer_m > 0:
            logger.warning(f"⚠️ Pas de pixel au point exact, searching dans buffer {buffer_m}m")
            feat = img.sample(region=pt.buffer(buffer_m), scale=scale, numPixels=1, geometries=False).first()
        
        if feat is None:
            logger.warning(f"⚠️ Impossible d'échantillonner un pixel même avec buffer")
            logger.warning(f"   → Utilisation d'embeddings simulés")
            return _generate_fallback_embedding(lat, lon, when)

        vals = feat.toDictionary().getInfo()

        # Extraire A01..A64 dans l'ordre
        bands = [f"A{i:02d}" for i in range(0, 64)]
        
        # Vérifier que tous les bands existent
        missing_bands = [b for b in bands if b not in vals]
        if missing_bands:
            logger.warning(f"⚠️ Bands manquants: {missing_bands}")
            logger.warning(f"   → Utilisation d'embeddings simulés")
            return _generate_fallback_embedding(lat, lon, when)
        
        embedding = [float(vals[b]) for b in bands]

        # Horodatage au format YYYY-MM-DD HH:MM:SS+00:00
        t0_ms = img.get("system:time_start").getInfo()
        if t0_ms is None:
            dt_ts = datetime(year, 1, 1, tzinfo=timezone.utc)
        else:
            dt_ts = datetime.fromtimestamp(t0_ms / 1000, tz=timezone.utc)
        
        time_utc = dt_ts.strftime("%Y-%m-%d %H:%M:%S+00:00")

        data = {
            "embedding": embedding,
            "lat": float(lat),
            "lon": float(lon),
            "time_utc": time_utc,
            "source": "AlphaEarth"
        }
        
        logger.info(f"✅ Embedding obtenu via AlphaEarth pour ({lat}, {lon})")
        return data

    except Exception as e:
        logger.warning(f"⚠️ Erreur lors du fetch AlphaEarth: {e}")
        logger.warning(f"   → Utilisation d'embeddings simulés")
        return _generate_fallback_embedding(lat, lon, when)


def _generate_fallback_embedding(lat: float, lon: float, when: str):
    """
    Génère un embedding de fallback réaliste quand AlphaEarth n'a pas de données.
    
    Cet embedding est:
    - Déterministe (même coordonnées = même embedding)
    - Réaliste (distribution normale)
    - Basé sur la localisation (latitude/longitude influencent l'embedding)
    """
    try:
        dt = datetime.fromisoformat(when)
    except:
        dt = datetime.now(timezone.utc)
    
    # Seed déterministe basé sur les coordonnées + date
    seed = int((lat * 1000 + lon * 1000 + dt.timetuple().tm_yday) * 1000) % (2**31)
    np.random.seed(seed)
    
    # Générer embedding 64D réaliste
    # Distribution normale avec variations basées sur la localisation
    embedding = np.random.randn(64).astype(float)
    
    # Normaliser légèrement basé sur latitude (tropical regions différent)
    lat_factor = 1.0 + (abs(lat) / 90.0) * 0.3
    embedding = (embedding * lat_factor).tolist()
    
    # Format timestamp au format attendu
    time_utc = dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    
    data = {
        "embedding": embedding,
        "lat": float(lat),
        "lon": float(lon),
        "time_utc": time_utc,
        "source": "Fallback"  # ← Indicate que c'est simulé
    }
    
    logger.info(f"📊 Embedding simulé pour ({lat}, {lon}) - source: Fallback")
    return data