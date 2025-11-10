import ee
from datetime import datetime, timezone


project="gen-lang-client-0546266030"

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
    """
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
    if col.size().getInfo() == 0:
        raise RuntimeError(f"Aucune tuile AlphaEarth ne couvre ({lat}, {lon}) en {year}.")

    img = col.first()

    # Pixel au point ; sinon on cherche dans un léger buffer
    feat = img.sample(region=pt, scale=scale, numPixels=1, geometries=False).first()
    if (feat is None) and buffer_m and buffer_m > 0:
        feat = img.sample(region=pt.buffer(buffer_m), scale=scale, numPixels=1, geometries=False).first()
    if feat is None:
        raise RuntimeError("Impossible d'échantillonner un pixel (NoData au point et dans le buffer).")

    vals = feat.toDictionary().getInfo()

    # Extraire A01..A64 dans l'ordre
    bands = [f"A{i:02d}" for i in range(0, 64)]
    embedding = [float(vals[b]) for b in bands]

    # Horodatage (début de période de l'image) en UTC avec suffixe +00:00
    t0_ms = img.get("system:time_start").getInfo()
    if t0_ms is None:
        # Par sécurité, fallback sur le 1er janvier de l'année
        t0_iso = datetime(year, 1, 1, tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    else:
        t0_iso = datetime.fromtimestamp(t0_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    # Insérer les deux-points dans le fuseau pour avoir ...+00:00
    time_utc = t0_iso[:-2] + ":" + t0_iso[-2:]

    # Construire l'objet EXACTEMENT comme demandé
    data_example = {
        "embedding": embedding,
        "lat": float(lat),
        "lon": float(lon),
        "time_utc": time_utc,
    }
    return data_example




