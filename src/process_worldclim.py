import rasterio
import numpy as np
from pathlib import Path
import glob

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
    
    print("Found monthly tavg files:")
    for f in monthly_files:
        print(f"  {f.name}")
    
    # Read the first file to get metadata
    with rasterio.open(monthly_files[0]) as src:
        meta = src.meta.copy()
        # We'll update the meta for the output (same as input but we are computing mean)
        # No change in dimensions, etc.
    
    # Initialize an array to accumulate the sum
    # We'll read the first band of each file and sum them.
    # We assume all files have the same dimensions and CRS.
    sum_array = None
    for f in monthly_files:
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            # Apply scale factor and offset for WorldClim tavg?
            # According to WorldClim 2.1 documentation, the tavg files are in degrees Celsius * 10 (i.e., integer values representing tenths of a degree).
            # We need to convert to degrees Celsius by dividing by 10.
            # Check the metadata: usually, there is no scale factor, but the values are stored as integers representing *10.
            # Let's check: if the data type is integer, we divide by 10. If it's float, we assume it's already in Celsius.
            # We'll check the data type and adjust.
            if np.issubdtype(data.dtype, np.integer):
                data = data / 10.0
            # If it's float, we assume it's already in Celsius (though WorldClim usually provides integer scaled by 10).
            # For safety, we can check the valid range, but we'll assume the user has the standard product.
            
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

if __name__ == "__main__":
    worldclim_dir = Path("data/WorldClim")
    if not worldclim_dir.exists():
        print("Error: WorldClim directory not found at data/WorldClim/")
        print("Please download the WorldClim 2.1 tavg_10m monthly dataset and extract it to data/WorldClim/")
        exit(1)
    
    try:
        bio1, meta = compute_worldclim_bio1(worldclim_dir)
        print("Bio1 computation successful!")
        print(f"Bio1 shape: {bio1.shape}")
        print(f"Bio1 min: {np.nanmin(bio1):.2f} °C")
        print(f"Bio1 max: {np.nanmax(bio1):.2f} °C")
        print(f"Bio1 mean: {np.nanmean(bio1):.2f} °C")
        
        # Save the Bio1 as a GeoTIFF
        output_path = Path("results/worldclim_bio1.tif")
        output_path.parent.mkdir(exist_ok=True)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(bio1.astype(rasterio.float32), 1)
        print(f"Saved Bio1 to: {output_path}")
        
    except Exception as e:
        print(f"Error processing WorldClim data: {e}")
        exit(1)
