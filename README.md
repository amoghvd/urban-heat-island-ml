# Urban Heat Island Analysis using Machine Learning

This project aims to build an end-to-end machine learning solution to predict urban heat island intensity using satellite imagery and socioeconomic data.

## Data Sources

1. **Landsat 8 Collection 2 Level-2** (from USGS EarthExplorer)
   - Required bands: ST_B10 (Surface Temperature), QA_PIXEL (Quality Assurance), SR_B4 (Red), SR_B5 (NIR), SR_B6 (SWIR1), SR_B7 (SWIR2)
   - Example: `LC08_L2SP_038037_20220615_20220628_02_T1`

2. **SEDAC GPWv4 Population Density** (2020, 30 arc-second)
   - File: `gpw_v4_population_density_rev11_2020_30_sec.tif`

3. **WorldClim 2.1** (10-minute resolution, monthly average temperature)
   - Files: `wc2.1_10m_tavg_01.tif` to `wc2.1_10m_tavg_12.tif` (representing January to December)

4. **OpenStreetMap** (extract for Arizona)
   - File: `arizona-latest.osm.pbf` (used for green/blue space features, though not fully implemented in the current script)

## Setup

1. Install required Python packages:
   ```
   pip install rasterio numpy
   ```

2. Download and prepare the data:
   - Landsat: Download a Level-2 (L2SP) scene for Phoenix, Arizona from USGS EarthExplorer (acquisition date in summer, e.g., 2022-06-15) with cloud cover <10%. Extract the `.tar.gz` file and place the extracted folder (containing the `.TIF` files) in `data/Landsat/`.
   - SEDAC: Download the GPWv4 Population Density 2020 (30 arc-second) GeoTIFF from [SEDAC](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4) or via [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/sedac) and place it in `data/SEDAC/` as `gpw_v4_population_density_rev11_2020_30_sec.tif`.
   - WorldClim: Download the WorldClim 2.1 10-minute resolution monthly average temperature (tavg) from [WorldClim](https://www.worldclim.org/data/worldclim21.html). Extract the 12 `.tif` files (one per month) and place them in `data/WorldClim/`.
   - OSM: Download the Arizona OSM extract from [Geofabrik](https://download.geofabrik.de/north-america/us/arizona-latest.osm.pbf) and place it in `data/OSM/` (though not used in the current feature engineering).

## Running the Analysis

Execute the main script:

```
python src/urban_heat_island_analysis.py
```

The script will:
1. Process the Landsat data to compute Land Surface Temperature (LST) in Celsius, mask clouds, and extract surface reflectance bands.
2. Compute the WorldClim Bio1 (Annual Mean Temperature) from the 12 monthly tavg files.
3. Engineer features by combining:
   - LST (from Landsat)
   - NDVI (Normalized Difference Vegetation Index)
   - NDBI (Normalized Difference Built-up Index)
   - Brightness (Red + NIR)/2
   - Population Density (from SEDAC, resampled to Landsat grid)
   - Bio1 (from WorldClim, resampled to Landsat grid)
4. Save the engineered features as a multi-band GeoTIFF in `results/features.tif`.
5. Save a list of feature names in `results/feature_names.txt`.
6. Print statistics for each feature.

## Output

- `results/features.tif`: A 6-band GeoTIFF with the following bands:
  1. LST (Celsius)
  2. NDVI
  3. NDBI
  4. Brightness
  5. Population Density (persons/km²)
  6. Bio1 (Annual Mean Temperature, Celsius)
- `results/feature_names.txt`: Text file listing the band names in order.

## Notes

- The script assumes that the Landsat data is in a subdirectory of `data/Landsat/` (e.g., `data/Landsat/LC08_L2SP_...`).
- Cloud masking is applied using the QA_PIXEL band (masking fill, dilated cloud, cirrus, cloud, and cloud shadow).
- All features are masked to NaN where clouds are present in the Landsat image.
- The WorldClim Bio1 is computed as the mean of the 12 monthly tavg files (after converting from tenths of degrees to degrees Celsius).
- The SEDAC population density and WorldClim Bio1 are resampled to match the Landsat grid using bilinear interpolation.

## Next Steps for Machine Learning

Once you have the `features.tif` file, you can:
1. Read it with rasterio or GDAL.
2. Flatten the arrays and mask out NaN values (clouds).
3. Split into training and test sets (consider spatial blocking to avoid autocorrelation).
4. Train models (e.g., Random Forest, XGBoost) to predict LST anomaly or extreme heat events.
5. Evaluate and interpret the results.

## References

- Landsat Collection 2 Level-2 Product Guide: https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-level-2-science-products
- GPWv4 Documentation: https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/documentation
- WorldClim 2.1: https://www.worldclim.org/data/worldclim21.html
- OpenStreetMap: https://www.openstreetmap.org

## License

This project is for educational purposes. Please refer to the respective data sources for their usage policies.
