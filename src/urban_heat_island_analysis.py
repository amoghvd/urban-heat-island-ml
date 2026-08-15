import rasterio
import numpy as np
from pathlib import Path
import sys

def process_landsat_landsat8L2(landsat_dir):
    """
    Process Landsat 8 Collection 2 Level-2 data to get:
    - Land Surface Temperature (LST) in Celsius
    - Cloud mask from QA_PIXEL
    - Surface Reflectance bands (SR_B4, SR_B5, SR_B6, SR_B7)

    Parameters:
    landsat_dir (str or Path): Directory containing the extracted Landsat L2SP files.

    Returns:
    dict: Dictionary containing the processed arrays and metadata.
    """
    landsat_path = Path(landsat_dir)

    # Define expected file names (we'll search for them)
    expected_files = {
        'ST_B10': '*ST_B10.TIF',
        'QA_PIXEL': '*QA_PIXEL.TIF',
        'SR_B4': '*SR_B4.TIF',
        'SR_B5': '*SR_B5.TIF',
        'SR_B6': '*SR_B6.TIF',
        'SR_B7': '*SR_B7.TIF'
    }

    # Find the files
    found_files = {}
    for key, pattern in expected_files.items():
        matches = list(landsat_path.glob(pattern))
        if len(matches) == 1:
            found_files[key] = matches[0]
        else:
            raise FileNotFoundError(
                f"Expected exactly one file matching {pattern} in {landsat_path}. "
                f"Found {len(matches)} files: {matches}"
            )

    # Open ST_B10 (Surface Temperature Band 10) and QA_PIXEL
    with rasterio.open(found_files['ST_B10']) as src_st:
        st_b10 = src_st.read(1).astype(np.float32)
        meta = src_st.meta.copy()
        # For ST_B10 in Landsat Collection 2 Level-2, the values are in Kelvin * 0.00341802 + 149.0
        # Convert to Kelvin first
        st_b10_kelvin = st_b10 * 0.00341802 + 149.0
        # Then convert to Celsius
        lst_celsius = st_b10_kelvin - 273.15

    with rasterio.open(found_files['QA_PIXEL']) as src_qa:
        qa_pixel = src_qa.read(1)
        # Landsat Collection 2 Level-2 QA_PIXEL bit mask:
        #   Bit 0: Fill (1 = fill)
        #   Bit 1: Dilated Cloud (1 = cloud)
        #   Bit 2: Cirrus (1 = cirrus)
        #   Bit 3: Cloud (1 = cloud)
        #   Bit 4: Cloud Shadow (1 = cloud shadow)
        #   Bit 5: Snow (1 = snow)
        #   Bit 6: Ice (1 = ice)
        # We want to mask: Fill, Dilated Cloud, Cirrus, Cloud, Cloud Shadow
        fill_mask = qa_pixel & 1
        dilated_cloud_mask = qa_pixel & 2
        cirrus_mask = qa_pixel & 4
        cloud_mask = qa_pixel & 8
        cloud_shadow_mask = qa_pixel & 16
        bad_mask = fill_mask | dilated_cloud_mask | cirrus_mask | cloud_mask | cloud_shadow_mask
        cloudmask = bad_mask.astype(np.bool_)

    # Apply cloud mask to LST
    lst_celsius[cloudmask] = np.nan

    # Now read the surface reflectance bands
    sr_bands = {}
    for band in ['B4', 'B5', 'B6', 'B7']:
        key = f'SR_{band}'
        with rasterio.open(found_files[key]) as src:
            arr = src.read(1).astype(np.float32)
            # Apply scale factor: Collection 2 Level-2 SR bands are scaled by 0.0000275 and offset -0.2
            arr = arr * 0.0000275 - 0.2
            # Apply cloud mask (and also mask fill values? We'll use the same cloudmask for consistency)
            sr_bands[band] = arr
            sr_bands[band][cloudmask] = np.nan

    # Update metadata for output
    meta.update({
        'dtype': rasterio.float32,
        'count': 1,
        'compress': 'lzw'
    })

    result = {
        'lst_celsius': lst_celsius,
        'cloudmask': cloudmask,
        'sr_bands': sr_bands,
        'meta': meta
    }

    return result

def compute_worldclim_bio1(worldclim_dir):
    """
    Compute WorldClim Bio1 (Annual Mean Temperature) from the 12 monthly average temperature (tavg) files.

    Parameters:
    worldclim_dir (str or Path): Directory containing the WorldClim tavg monthly files (wc2.1_10m_tavg_01.tif to wc2.1_10m_tavg_12.tif).

    Returns:
    tuple: (bio1_array, meta) where bio1_array is the annual mean temperature in degrees Celsius, and meta is the metadata profile.
    """
    worldclim_path = Path(worldclim_dir)

    # List the monthly tavg files
    monthly_files = sorted(worldclim_path.glob("wc2.1_10m_tavg_*.tif"))
    if len(monthly_files) != 12:
        raise FileNotFoundError(
            f"Expected 12 monthly tavg files in {worldclim_path}, found {len(monthly_files)}. "
            "Please ensure you have downloaded the WorldClim 2.1 tavg_10m monthly dataset."
        )

    # Read the first file to get metadata
    with rasterio.open(monthly_files[0]) as src:
        meta = src.meta.copy()

    # Initialize an array to accumulate the sum
    sum_array = None
    for f in monthly_files:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            # WorldClim tavg files are in degrees Celsius * 10 (i.e., integer values representing tenths of a degree)
            if np.issubdtype(data.dtype, np.integer):
                data = data / 10.0
            # If it's float, we assume it's already in Celsius (though uncommon)

            if sum_array is None:
                sum_array = np.zeros_like(data, dtype=np.float64)
            sum_array += data

    # Compute the mean
    bio1_array = sum_array / len(monthly_files)

    # Update metadata for output
    meta.update({
        'dtype': rasterio.float32,
        'count': 1,
        'compress': 'lzw'
    })

    return bio1_array, meta

def resample_to_match(source_path, destination_meta):
    """
    Resample a source raster to match the destination meta (transform, width, height, crs).

    Parameters:
    source_path (str or Path): Path to the source raster.
    destination_meta (dict): The metadata of the destination raster (transform, width, height, crs).

    Returns:
    np.ndarray: The resampled array.
    """
    from rasterio.warp import reproject, Resampling
    with rasterio.open(source_path) as src:
        dest_arr = np.zeros((destination_meta['height'], destination_meta['width']), dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, 1),
            dest=dest_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=destination_meta['transform'],
            dst_crs=destination_meta['crs'],
            resampling=Resampling.bilinear
        )
    return dest_arr

def main():
    print("=== Urban Heat Island Feature Engineering ===")
    print("Step 1: Checking data...")

    # Check Landsat directory
    landsat_base = Path("data/Landsat")
    if not landsat_base.exists():
        print("ERROR: Landsat directory not found at data/Landsat/")
        sys.exit(1)
    scene_dirs = [d for d in landsat_base.iterdir() if d.is_dir()]
    if len(scene_dirs) == 0:
        print("ERROR: No scene directory found in data/Landsat/")
        print("Please download and extract a Landsat 8 Collection 2 Level-2 scene (L2SP) into data/Landsat/")
        print("Required bands: ST_B10, QA_PIXEL, SR_B4, SR_B5, SR_B6, SR_B7")
        sys.exit(1)
    elif len(scene_dirs) > 1:
        print(f"WARNING: Multiple scene directories found in data/Landsat/: {scene_dirs}")
        print("Using the first one.")
    landsat_dir = scene_dirs[0]
    print(f"  Using Landsat scene: {landsat_dir.name}")

    # Check SEDAC population density
    sedac_path = Path("data/SEDAC/gpw_v4_population_density_rev11_2020_30_sec.tif")
    if not sedac_path.exists():
        print("ERROR: SEDAC population density file not found at data/SEDAC/gpw_v4_population_density_rev11_2020_30_sec.tif")
        print("Please download the SEDAC GPWv4 Population Density 2020 (30 arc-second) GeoTIFF.")
        sys.exit(1)
    print("  SEDAC population density: OK")

    # Check WorldClim directory
    worldclim_base = Path("data/WorldClim")
    if not worldclim_base.exists():
        print("ERROR: WorldClim directory not found at data/WorldClim/")
        sys.exit(1)
    monthly_files = list(worldclim_base.glob("wc2.1_10m_tavg_*.tif"))
    if len(monthly_files) != 12:
        print("ERROR: Expected 12 monthly tavg files in data/WorldClim/ (wc2.1_10m_tavg_01.tif to wc2.1_10m_tavg_12.tif)")
        print("Please download the WorldClim 2.1 tavg_10m monthly dataset and extract it to data/WorldClim/")
        sys.exit(1)
    print("  WorldClim monthly tavg files: OK (12 files found)")

    print("\nStep 2: Processing Landsat data...")
    try:
        landsat_result = process_landsat_landsat8L2(landsat_dir)
        print("  Landsat processing successful!")
        print(f"    LST shape: {landsat_result['lst_celsius'].shape}")
        print(f"    LST min: {np.nanmin(landsat_result['lst_celsius']):.2f} °C")
        print(f"    LST max: {np.nanmax(landsat_result['lst_celsius']):.2f} °C")
        print(f"    LST mean: {np.nanmean(landsat_result['lst_celsius']):.2f} °C")
    except Exception as e:
        print(f"  ERROR processing Landsat data: {e}")
        sys.exit(1)

    print("\nStep 3: Computing WorldClim Bio1...")
    try:
        bio1_array, bio1_meta = compute_worldclim_bio1(worldclim_base)
        print("  WorldClim Bio1 computation successful!")
        print(f"    Bio1 shape: {bio1_array.shape}")
        print(f"    Bio1 min: {np.nanmin(bio1_array):.2f} °C")
        print(f"    Bio1 max: {np.nanmax(bio1_array):.2f} °C")
        print(f"    Bio1 mean: {np.nanmean(bio1_array):.2f} °C")
    except Exception as e:
        print(f"  ERROR computing WorldClim Bio1: {e}")
        sys.exit(1)

    print("\nStep 4: Engineering features...")
    try:
        # Prepare output metadata based on Landsat
        out_meta = landsat_result['meta'].copy()
        out_meta.update({
            'count': 6,
            'dtype': rasterio.float32,
            'compress': 'lzw'
        })

        # Initialize output array
        out_shape = (out_meta['height'], out_meta['width'])
        feature_array = np.zeros((6, out_shape[0], out_shape[1]), dtype=np.float32)

        # Band 0: LST
        feature_array[0] = landsat_result['lst_celsius']

        # Band 1: NDVI = (NIR - Red) / (NIR + Red)
        nir = landsat_result['sr_bands']['B5']
        red = landsat_result['sr_bands']['B4']
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)
        feature_array[1] = ndvi

        # Band 2: NDBI = (SWIR - NIR) / (SWIR + NIR)
        swir = landsat_result['sr_bands']['B6']
        denominator = swir + nir
        ndbi = np.where(denominator != 0, (swir - nir) / denominator, np.nan)
        feature_array[2] = ndbi

        # Band 3: Brightness = (Red + NIR) / 2
        brightness = (red + nir) / 2.0
        feature_array[3] = brightness

        # Band 4: Population Density (from SEDAC)
        # Resample SEDAC to match Landsat grid
        pop_density = resample_to_match(sedac_path, out_meta)
        feature_array[4] = pop_density.astype(np.float32)
        # Mask where Landsat has clouds (so we don't use population in clouds for modeling)
        feature_array[4][landsat_result['cloudmask']] = np.nan

        # Band 5: WorldClim Bio1
        # We have the bio1_array and bio1_meta from the WorldClim processing.
        # We need to resample the bio1_array (which is in the WorldClim grid) to match the Landsat grid.
        # We'll use the bio1_array and bio1_meta to create a temporary in-memory raster and then resample.
        # Alternatively, we can write a temporary file, but let's do it in memory with reproject.
        from rasterio.warp import reproject, Resampling

        # Create a destination array for the resampled Bio1
        bio1_dest = np.zeros((out_meta['height'], out_meta['width']), dtype=bio1_array.dtype)
        # Reproject the bio1_array (which we have as a numpy array) to match out_meta
        # We need to treat bio1_array as a raster with bio1_meta
        reproject(
            source=bio1_array,
            dest=bio1_dest,
            src_transform=bio1_meta['transform'],
            src_crs=bio1_meta['crs'],
            dst_transform=out_meta['transform'],
            dst_crs=out_meta['crs'],
            resampling=Resampling.bilinear
        )
        feature_array[5] = bio1_dest.astype(np.float32)
        # Mask where Landsat has clouds
        feature_array[5][landsat_result['cloudmask']] = np.nan

        # Apply cloud mask to all features (so that clouds are NaN in all bands)
        for i in range(6):
            feature_array[i][landsat_result['cloudmask']] = np.nan

        # Save the multi-band GeoTIFF
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "features.tif"
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            for i in range(6):
                dst.write(feature_array[i], i+1)

        print(f"  Features saved to: {output_path}")
        print(f"    Number of bands: {out_meta['count']}")
        feature_names = ['LST', 'NDVI', 'NDBI', 'Brightness', 'PopDensity', 'Bio1']
        for i, name in enumerate(feature_names):
            print(f"    Band {i+1}: {name}")

        # Save feature names
        with open(output_dir / "feature_names.txt", 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"  Feature names saved to: {output_dir / 'feature_names.txt'}")

        # Print statistics for each feature (ignoring NaN)
        print("\nFeature statistics (mean, min, max) - ignoring NaN:")
        for i, name in enumerate(feature_names):
            valid_vals = feature_array[i][~np.isnan(feature_array[i])]
            if len(valid_vals) > 0:
                print(f"  {name}: mean={np.mean(valid_vals):.2f}, min={np.min(valid_vals):.2f}, max={np.max(valid_vals):.2f}")
            else:
                print(f"  {name}: all values are NaN")

    except Exception as e:
        print(f"  ERROR engineering features: {e}")
        sys.exit(1)

    print("\n=== Analysis Complete ===")
    print("The engineered features are saved in results/features.tif")
    print("You can now use this file for training your machine learning model.")

if __name__ == "__main__":
    main()