import os
import cdsapi
import xarray as xr
import pandas as pd


def download_and_extract_pblh():
    c = cdsapi.Client()

    # Obtiene la ruta exacta de la carpeta donde reside este script
    carpeta_script = os.path.dirname(os.path.abspath(__file__))

    archivo_nc = os.path.join(carpeta_script, "CapaDeMezcla.nc")
    archivo_csv = os.path.join(carpeta_script, "CapaDeMezcla.csv")

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['boundary_layer_height'],
            'year': '2023',
            'month': '03',
            'day': [f"{d:02d}" for d in range(12, 21)],
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': [4.80, -74.25, 4.50, -74.00],
        },
        archivo_nc
    )

    ds = xr.open_dataset(archivo_nc)
    point_series = ds['blh'].sel(
        latitude=4.63, longitude=-74.11, method='nearest'
    ).to_dataframe()
    point_series.reset_index(inplace=True)

    point_series['ZI'] = point_series['blh'].apply(
        lambda x: round(float(x), 1) if pd.notnull(x) else -999.0
    )
    point_series.to_csv(archivo_csv, index=False)

    print(f"NetCDF guardado en: {archivo_nc}")
    print(f"CSV guardado en:    {archivo_csv}")


if __name__ == '__main__':
    download_and_extract_pblh()