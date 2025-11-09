import ee
import numpy as np

ee.Initialize(project='gen-lang-client-0546266030')

# Définir un point simple (ici Paris)
pt = ee.Geometry.Point(2.35, 48.85)

# Filtrer la collection pour garder uniquement les images couvrant ce point
collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filterBounds(pt)

# Vérifie qu’il y a bien une image
count = collection.size().getInfo()
print("Nombre d'images couvrant le point :", count)

if count == 0:
    raise RuntimeError("Aucune tuile d'embedding ne couvre ce point.")

# Prendre la première image valide
img = collection.first()

# Échantillonner ce point
sample = img.sample(region=pt, scale=30).first()

if sample is None:
    raise RuntimeError("⚠️ Aucun pixel valide trouvé dans cette tuile (peut-être NoData).")

# Transformer en dictionnaire
vals = sample.toDictionary().getInfo()
embedding = np.array([vals[f"A{i:02d}"] for i in range(0, 64)], dtype=float)
print("Embedding vector (64):", embedding)
print("embedding value types:", [type(vals[f"A{i:02d}"]) for i in range(0, 64)])

print("Couverture de la tuile :", img.geometry().bounds(1).getInfo())
#                         ici ---------^   maxError en mètres (par ex. 1)

