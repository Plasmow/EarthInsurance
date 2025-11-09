import ee
ee.Initialize(project='gen-lang-client-0546266030')
embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')