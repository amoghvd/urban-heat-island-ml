import rasterio
import numpy as np
from pathlib import Path
import sys
from rasterio.warp import reproject, Resampling

def main():
    print("=== Urban Heat Island Feature Engineering ===")
    # Base directory: parent of script's directory (assuming script in src/)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    # Paths
    landsat_base = base_dir / "data/Landsat"
    # Find the scene directory (should be exactly one, exclude results)
    scene_dirs = [d for d in landsat_base.iterdir() if d.is_dir() and d.name != 'results']
    if len(scene_dirs) == 0:
        print("ERROR: No scene directory found in data/Landsat/")
        print("Please download and extract a Landsat 8 Collection 2 Level-2 scene (L2SP) into data/Landsat/")
        sys.exit(1)
    elif len(scene_dirs) > 1:
        print(f"WARNING: Multiple scene directories found in data/Landsat/: {scene_dirs}")
        print("Using the first one.")
    landsat_scene_dir = scene_dirs[0]
    scene_prefix = landsat_scene_dir.name  # e.g., LC08_L2SP_001069_20220615_20220627_02_T1
    print(f"Using Landsat scene: {scene_prefix}")

    sedac_path = base_dir / "data/SEDAC/gpw_v4_population_density_rev11_2020_30_sec_2020.tif"
    worldclim_dir = base_dir / "data/WorldClim"
    output_dir = base_dir / "results"
    output_dir.mkdir(exist_ok=True)

    # 1. Process Landsat
    print("Step 1: Processing Landsat data...")
    # Build file paths
    st_b10_path = landsat_scene_dir / f"{scene_prefix}_ST_B10.TIF"
    qa_pixel_path = landsat_scene_dir / f"{scene_prefix}_QA_PIXEL.TIF"
    sr_bands = {}
    for b in ['B4','B5','B6','B7']:
        sr_bands[b] = landsat_scene_dir / f"{scene_prefix}_SR_{b}.TIF"

    # Check existence
    for p in [st_b10_path, qa_pixel_path] + list(sr_bands.values()):
        if not p.exists():
            print(f"ERROR: Missing file: {p}")
            sys.exit(1)

    # Open ST_B10 and QA_PIXEL
    with rasterio.open(st_b10_path) as src_st:
        st_b10 = src_st.read(1).astype(np.float32)
        meta = src_st.meta.copy()
        # Mask nodata values
        if src_st.nodata is not None:
            st_b10 = np.where(st_b10 == src_st.nodata, np.nan, st_b10)
        # Convert to Kelvin
        st_b10_kelvin = st_b10 * 0.00341802 + 149.0
        lst_celsius = st_b10_kelvin - 273.15
    with rasterio.open(qa_pixel_path) as src_qa:
        qa_pixel = src_qa.read(1)
        # Mask: fill, dilated cloud, cirrus, cloud, cloud shadow
        fill_mask = qa_pixel & 1
        dilated_cloud_mask = qa_pixel & 2
        cirrus_mask = qa_pixel & 4
        cloud_mask = qa_pixel & 8
        cloud_shadow_mask = qa_pixel & 16
        bad_mask = fill_mask | dilated_cloud_mask | cirrus_mask | cloud_mask | cloud_shadow_mask
        cloudmask = bad_mask.astype(np.bool_)
    lst_celsius[cloudmask] = np.nan

    # Read SR bands
    sr_bands_data = {}
    for b, path in sr_bands.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            # Apply scale factor: Collection 2 Level-2 SR bands: *0.0000275 - 0.2
            arr = arr * 0.0000275 - 0.2
            arr[cloudmask] = np.nan
            sr_bands_data[b] = arr

    print(f"  LST shape: {lst_celsius.shape}")
    print(f"  LST min: {np.nanmin(lst_celsius):.2f} °C")
    print(f"  LST max: {np.nanmax(lst_celsius):.2f} °C")
    print(f"  LST mean: {np.nanmean(lst_celsius):.2f} °C")

    # 2. Compute WorldClim Bio1
    print("Step 2: Computing WorldClim Bio1...")
    monthly_files = sorted(worldclim_dir.glob("wc2.1_10m_tavg_*.tif"))
    if len(monthly_files) != 12:
        raise FileNotFoundError(f"Expected 12 monthly tavg files, found {len(monthly_files)}")
    meta_bio1 = None
    sum_array = None
    for f in monthly_files:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            if np.issubdtype(data.dtype, np.integer):
                data = data / 10.0  # Convert from tenths of deg C to deg C
            # Also mask unrealistic values (annual mean temp cannot be < -50 or > 50 °C)
            data = np.where((data < -50) | (data > 50), np.nan, data)
            if sum_array is None:
                sum_array = np.zeros_like(data, dtype=np.float64)
            sum_array += data
            if meta_bio1 is None:
                meta_bio1 = src.meta.copy()
    bio1_array = sum_array / len(monthly_files)
    meta_bio1.update({'dtype': rasterio.float32, 'count': 1, 'compress': 'lzw'})
    print(f"  Bio1 shape: {bio1_array.shape}")
    print(f"  Bio1 min: {np.nanmin(bio1_array):.2f} °C")
    print(f"  Bio1 max: {np.nanmax(bio1_array):.2f} °C")
    print(f"  Bio1 mean: {np.nanmean(bio1_array):.2f} °C")

    # 3. Engineer features
    print("Step 3: Engineering features...")
    out_meta = meta.copy()
    out_meta.update({
        'count': 6,
        'dtype': rasterio.float32,
        'compress': 'lzw'
    })
    out_shape = (out_meta['height'], out_meta['width'])
    feature_array = np.zeros((6, out_shape[0], out_shape[1]), dtype=np.float32)

    # Band 0: LST
    feature_array[0] = lst_celsius

    # Band 1: NDVI = (NIR - Red) / (NIR + Red)
    nir = sr_bands_data['B5']
    red = sr_bands_data['B4']
    denominator = nir + red
    ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)
    feature_array[1] = ndvi

    # Band 2: NDBI = (SWIR - NIR) / (SWIR + NIR)
    swir = sr_bands_data['B6']
    denominator = swir + nir
    ndbi = np.where(denominator != 0, (swir - nir) / denominator, np.nan)
    feature_array[2] = ndbi

    # Band 3: Brightness = (Red + NIR) / 2
    brightness = (red + nir) / 2.0
    feature_array[3] = brightness

    # Band 4: Population Density (from SEDAC)
    # Resample SEDAC to match Landsat grid
    with rasterio.open(sedac_path) as src_sedac:
        pop_density = np.zeros(out_shape, dtype=src_sedac.dtypes[0])
        reproject(
            source=rasterio.band(src_sedac, 1),
            destination=pop_density,
            src_transform=src_sedac.transform,
            src_crs=src_sedac.crs,
            dst_transform=out_meta['transform'],
            dst_crs=out_meta['crs'],
            resampling=Resampling.bilinear
        )
    feature_array[4] = pop_density.astype(np.float32)
    # Mask nodata from SEDAC
    with rasterio.open(sedac_path) as src_sedac_check:
        if src_sedac_check.nodata is not None:
            pop_density = np.where(pop_density == src_sedac_check.nodata, np.nan, pop_density)
    feature_array[4][cloudmask] = np.nan  # mask clouds

    # Band 5: WorldClim Bio1
    # Resample Bio1 to match Landsat grid
    bio1_resampled = np.zeros(out_shape, dtype=bio1_array.dtype)
    reproject(
        source=bio1_array,
        destination=bio1_resampled,
        src_transform=meta_bio1['transform'],
        src_crs=meta_bio1['crs'],
        dst_transform=out_meta['transform'],
        dst_crs=out_meta['crs'],
        resampling=Resampling.bilinear
    )
    feature_array[5] = bio1_resampled.astype(np.float32)
    # Mask nodata from Bio1 (already done in bio1_array, but ensure)
    with rasterio.open(worldclim_dir / "wc2.1_10m_tavg_01.tif") as src_bio1_check:
        if src_bio1_check.nodata is not None:
            bio1_resampled = np.where(bio1_resampled == src_bio1_check.nodata, np.nan, bio1_resampled)
    feature_array[5][cloudmask] = np.nan

    # Apply cloud mask to all features (already done for some, but ensure)
    for i in range(6):
        feature_array[i][cloudmask] = np.nan

    # Save multi-band GeoTIFF
    output_path = base_dir / "results" / "features.tif"
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        for i in range(6):
            dst.write(feature_array[i], i+1)

    # Save feature names
    feature_names = ['LST', 'NDVI', 'NDBI', 'Brightness', 'PopDensity', 'Bio1']
    with open(base_dir / "results" / "feature_names.txt", 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")

    print(f"  Features saved to: {output_path}")
    print(f"    Number of bands: {out_meta['count']}")
    for i, name in enumerate(feature_names):
        print(f"    Band {i+1}: {name}")

    # Print statistics
    print("\nFeature statistics (mean, min, max) - ignoring NaN:")
    for i, name in enumerate(feature_names):
        valid_vals = feature_array[i][~np.isnan(feature_array[i])]
        if len(valid_vals) > 0:
            print(f"  {name}: mean={np.mean(valid_vals):.2f}, min={np.min(valid_vals):.2f}, max={np.max(valid_vals):.2f}")
        else:
            print(f"  {name}: all values are NaN")

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)