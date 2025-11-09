import ee
import numpy as np

ee.Initialize(project='gen-lang-client-0546266030')

# Define a simple point (Paris)
pt = ee.Geometry.Point(2.35, 48.85)

# Filter the collection to keep only images covering this point
collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filterBounds(pt)

# Ensure at least one image is available
count = collection.size().getInfo()
print("Number of images covering the point:", count)

if count == 0:
    raise RuntimeError("No embedding tile covers this point.")

# Take the first valid image
img = collection.first()

# Sample this point
sample = img.sample(region=pt, scale=30).first()

if sample is None:
    raise RuntimeError("⚠️ No valid pixel found in this tile (possibly NoData).")

# Convert to a dictionary
vals = sample.toDictionary().getInfo()
embedding = np.array([vals[f"A{i:02d}"] for i in range(0, 64)], dtype=float)
print("Embedding vector (64):", embedding)
print("embedding value types:", [type(vals[f"A{i:02d}"]) for i in range(0, 64)])

print("Tile coverage:", img.geometry().bounds(1).getInfo())
#                            here -----^   maxError in meters (e.g., 1)

