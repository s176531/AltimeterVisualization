import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import xarray as xr
import numpy.typing as npt
import cartopy.crs as ccrs
from pathlib import Path
from typing import Tuple, List, Iterable, Callable
import multiprocessing
import moviepy.video.io.ImageSequenceClip as ISC


def make_png(
        image: npt.ArrayLike,
        figsize: Tuple[int, int],
        output_path: Path,
        extent: List[int] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = 'jet',
        title: str | None = None,
        cbar_label: str | None = None,
    ):
    """Makes a png based of a grid"""
    if extent is None:
        extent = [-180, 180, -90, 90]
    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max()

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
    im = ax.imshow(image, origin='lower', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)

    ax.coastlines()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.set_xticks([-180,-120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax.set_xlabel(f"Longitude [\N{DEGREE SIGN}]")
    ax.set_ylabel(f"Latitude [\N{DEGREE SIGN}]")

    if title is not None:
        ax.set_title(title)
    
    plt.savefig(output_path, format='png')
    plt.close()


def rotate_data(data: xr.DataArray, coord: str, upper_boundary: float):
    # Assign lower bound
    bool_arr = data[coord] < upper_boundary
    if bool_arr.sum().item() == len(data[coord]):
        return data.isel(**{coord: bool_arr}).values
    new_data = np.empty(data.shape)
    new_data[:, ~bool_arr] = data.isel(**{coord: bool_arr})
    new_data[:, bool_arr] = data.isel(**{coord: ~bool_arr})
    return new_data

def processed_to_date(file: Path) -> date:
    year, month, day = file.name.replace('.nc', '').replace('.png', '').split('_')
    return date(int(year), int(month), int(day))

def cmems_to_date(file: Path) -> date:
    cleaned_filename = file.name.replace('.nc', '').replace('.png', '').replace('dt_global_allsat_phy_l4_', '')
    if 'vDT2021' in cleaned_filename:
        date_name = cleaned_filename.split('_')[-2]
    else:
        date_name = cleaned_filename.split('_')[0]
    return date(int(date_name[:4]), int(date_name[4:6]), int(date_name[6:]))

def measures_to_date(file: Path) -> date:
    date_name = file.name.replace('.nc', '').replace('.png', '').replace('ssh_grids_v2205_', '')
    return date(int(date_name[:4]), int(date_name[4:6]), int(date_name[6:8]))






def _multiprocessing_processed_to_pngs(grid: Path, output_folder: Path, feature: str = 'sla'):
    image = xr.open_dataset(grid)
    make_png(
        image = image[feature],
        figsize = (15, 5),
        output_path = output_folder / f"{grid.name.replace('.nc', '')}.png",
        extent = [-180, 180, -90, 90],
        vmin = -0.75,
        vmax = 0.75,
        title = f"Ours - {processed_to_date(grid)} - Time weighting = 0.1"
    )

def _multiprocessing_cmems_to_pngs(grid: Path, output_folder: Path, feature: str = 'sla'):
    image = xr.open_dataset(grid, engine='netcdf4')
    new_cmems = rotate_data(image[feature][0], 'longitude', 180)
    make_png(
        image = new_cmems,
        figsize = (15, 5),
        output_path = output_folder / f"{grid.name.replace('.nc', '')}.png",
        extent = [-180, 180, -90, 90],
        vmin = -0.75,
        vmax = 0.75,
        title = f"CMEMS - {cmems_to_date(grid)}"
    )

def _multiprocessing_measures_to_pngs(grid: Path, output_folder: Path, feature: str = 'sla'):
    image = xr.open_dataset(grid)
    new_measures = rotate_data(image[feature][0], 'Longitude', 180)
    new_measures = np.pad(new_measures, ((60, 60), (0,0)), mode='constant', constant_values=(np.nan,))
    make_png(
        image = new_measures,
        figsize = (15, 5),
        output_path = output_folder / f"{grid.name.replace('.nc', '')}.png",
        extent = [-180, 180, -90, 90],
        vmin = -0.75,
        vmax = 0.75,
        title = f"MEaSUREs - {measures_to_date(grid)}"
    )


def folder_to_mp4(folder: Path, func: Callable[[Path], date], fps: int, file_name: str):
    dates = [(func(file), file) for file in folder.glob('*.png')]
    sorted_by_dates = sorted(dates, key=lambda tup: tup[0])
    files = [d[1].as_posix() for d in sorted_by_dates]
    clip = ISC.ImageSequenceClip(files, fps)
    clip.write_videofile((folder / file_name).as_posix())
 

def processed_main():
    grids_folder = Path(r"C:\Users\Casper\OneDrive - Danmarks Tekniske Universitet\SKOLE\Kandidat\Syntese\ProcessedGrids\Processed_v4_v10")
    output_folder = Path("Images", "TD01_2019")
    feature = 'sla'

    grids = grids_folder.glob('2019*.nc')
    output_folder.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool() as pool:
        _ = pool.starmap(_multiprocessing_processed_to_pngs, [(grid, output_folder, feature) for grid in grids])
    
    folder_to_mp4(output_folder, processed_to_date, 15, "2019_ours.mp4")

def cmems_main():
    grids_folder = Path(r"C:\Users\Casper\OneDrive - Danmarks Tekniske Universitet\SKOLE\Kandidat\Syntese\ProcessedGrids\CMEMS")
    output_folder = Path("Images", "CMEMS_2019")
    feature = 'sla'

    grids = grids_folder.glob('*_2019[0-9][0-9][0-9][0-9]_*.nc')
    output_folder.mkdir(parents=True, exist_ok=True)

    # for grid in grids:
    #     _multiprocessing_cmems_to_pngs(grid, output_folder, feature)

    with multiprocessing.Pool() as pool:
        _ = pool.starmap(_multiprocessing_cmems_to_pngs, [(grid, output_folder, feature) for grid in grids])
    
    folder_to_mp4(output_folder, cmems_to_date, 15, "2019_CMEMS.mp4")

def measures_main():
    grids_folder = Path(r"C:\Users\Casper\OneDrive - Danmarks Tekniske Universitet\SKOLE\Kandidat\Syntese\ProcessedGrids\MEaSUREs")
    output_folder = Path("Images", "MEaSUREs_2019")
    feature = 'SLA'

    grids = grids_folder.glob('*2019*.nc')
    output_folder.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool() as pool:
        _ = pool.starmap(_multiprocessing_measures_to_pngs, [(grid, output_folder, feature) for grid in grids])
    
    folder_to_mp4(output_folder, measures_to_date, 15, "2019_MEaSUREs.mp4")

if __name__ == '__main__':
    #processed_main()
    cmems_main()
    #measures_main()