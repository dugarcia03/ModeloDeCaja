import cdsapi
import xarray as xr
import pandas as pd


def download_and_extract_pblh():
    c = cdsapi.Client()

    # Descarga ERA5 reanálisis horario del 13-03-2023 al 20-03-2023
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['boundary_layer_height'],
            'year': '2023',
            'month': '03',
            'day': [f"{d:02d}" for d in range(13, 21)],  # Días 13 al 20 inclusive
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': [4.80, -74.25, 4.50, -74.00],  # Bounding Box Bogotá
        },
        'era5_pblh_bogota_20230313_20230320.nc'
    )

    # Apertura y selección de la coordenada geográfica de Puente Aranda
    ds = xr.open_dataset('era5_pblh_bogota_20230313_20230320.nc')
    point_series = ds['blh'].sel(
        latitude=4.63, longitude=-74.11, method='nearest'
    ).to_dataframe()
    point_series.reset_index(inplace=True)

    # Limpieza y asignación de la variable ZI (m)
    point_series['ZI'] = point_series['blh'].apply(
        lambda x: round(float(x), 1) if pd.notnull(x) else -999.0
    )
    point_series.to_csv('era5_pblh_puente_aranda_20230313_20230320.csv', index=False)
    print("Archivo 'era5_pblh_puente_aranda_20230313_20230320.csv' generado correctamente.")


if __name__ == '__main__':
    download_and_extract_pblh()