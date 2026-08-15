import rasterio
import numpy as np
import os
from pathlib import Path

def process_landsat_landsat8L2(landsat_dir):
    """
    Process Landsat 8 Collection 2 Level-2 data to get:
    - Land Surface Temperature (LST) in Celsius
    - Cloud mask from QA_PIXEL
    - Surface Reflectance bands (SR_B4, SR_B5, SR_B6, SR_B7) for indices
    
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
    
    print("Found Landsat files:")
    for key, path in found_files.items():
        print(f"  {key}: {path.name}")
    
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

if __name__ == "__main__":
    # Example usage: process the Landsat data in data/Landsat/<scene_folder>
    # We expect the user to have placed the extracted Landsat L2SP folder in data/Landsat/
    landsat_base = Path("data/Landsat")
    # Find the first directory that looks like a Landsat scene (we assume only one)
    scene_dirs = [d for d in landsat_base.iterdir() if d.is_dir()]
    if len(scene_dirs) == 0:
        print("Error: No scene directory found in data/Landsat/")
        print("Please download and extract a Landsat 8 Collection 2 Level-2 scene (L2SP) into data/Landsat/")
        print("Required bands: ST_B10, QA_PIXEL, SR_B4, SR_B5, SR_B6, SR_B7")
        exit(1)
    elif len(scene_dirs) > 1:
        print(f"Warning: Multiple scene directories found in data/Landsat/: {scene_dirs}")
        print("Using the first one.")
    
    landsat_dir = scene_dirs[0]
    print(f"Processing Landsat scene in: {landsat_dir}")
    
    try:
        result = process_landsat_landsat8L2(landsat_dir)
        print("Processing successful!")
        print(f"LST shape: {result['lst_celsius'].shape}")
        print(f"LST min: {np.nanmin(result['lst_celsius']):.2f} °C")
        print(f"LST max: {np.nanmax(result['lst_celsius']):.2f} °C")
        print(f"LST mean: {np.nanmean(result['lst_celsius']):.2f} °C")
        
        # Save the LST as a GeoTIFF for inspection
        output_path = Path("results/phoenix_lst_jun2022.tif")
        output_path.parent.mkdir(exist_ok=True)
        with rasterio.open(output_path, 'w', **result['meta']) as dst:
            dst.write(result['lst_celsius'].astype(rasterio.float32), 1)
        print(f"Saved LST to: {output_path}")
        
    except Exception as e:
        print(f"Error processing Landsat data: {e}")
        exit(1)
