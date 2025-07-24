import os
import json
import requests
import geopandas as gpd
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

from typing import Union

root = "C:/Users/CND905/Downloaded_Programs/chmdata/src"  # <--- Change to your location
os.chdir(root)
import chmdata

def get_station_locations(
        geometry: Union[Path, gpd.GeoDataFrame]
        ) -> gpd.GeoDataFrame:
    '''
    Queries mesonet and agrimet stations using polygon geometry
    :param geometry: Path or geopandas.GeoDataFrame - polygon geometry of area that stations will be extracted
    :return: geopandas.GeoDataFrame of stations within polygon area
    '''
    if geometry is not None:
        if isinstance(geometry, (str, Path)):
            in_geom = gpd.read_file(geometry)
        elif isinstance(geometry, gpd.GeoDataFrame):
            in_geom = geometry
        else:
            raise ValueError("The input geometry is neither a string path nor a geopandas GeoDataFrame.")

    adf = get_agrimet_locations(geometry=in_geom)
    mdf = get_mesonet_locations(geometry=in_geom)
    df = pd.concat([adf, mdf]).reset_index(drop=True)

    crs = 4326
    if df.crs is None:
        df = df.set_crs(crs)
    else:
        df = df.to_crs(crs)

    if in_geom is None:
        in_geom = in_geom.set_crs(crs)
    else:
        in_geom = in_geom.to_crs(crs)

    fig, ax = plt.subplots()
    in_geom.boundary.plot(ax=ax)
    df.plot(column='network', ax=ax, legend=True, cmap='Accent')

    return df

def get_agrimet_locations(
        geometry: Union[Path, gpd.GeoDataFrame]
        ) -> gpd.GeoDataFrame:
    '''
    Queries agrimet stations using polygon geometry
    :param geometry: Path or geopandas.GeoDataFrame - polygon geometry of area that stations will be extracted
    :return: geopandas.GeoDataFrame of station with metadata
    '''
    agrimet_stns = chmdata.agrimet.load_stations()
    stn_list = list(agrimet_stns.keys())
    dic_list = []
    for stn in stn_list:
        dic_list.append(agrimet_stns[stn])

    gdf_return = gpd.GeoDataFrame.from_features(dic_list)
    fin_gdf = gdf_return

    if geometry is not None:
        if isinstance(geometry, (str, Path)):
            in_geom = gpd.read_file(geometry)
        elif isinstance(geometry, gpd.GeoDataFrame):
            in_geom = geometry
        else:
            raise ValueError("The input geometry is neither a string path nor a geopandas GeoDataFrame.")

        in_geom = in_geom.to_crs(4326)
        fin_gdf = fin_gdf.loc[fin_gdf.intersects(in_geom.geometry[0]), :]

    fin_gdf = gpd.GeoDataFrame(
        {'name': fin_gdf['siteid'].to_list(),
        'network': ['agrimet'] * len(fin_gdf),
        'sub_network': fin_gdf['type'].to_list(),
        'install': pd.to_datetime(fin_gdf['install']).to_list(),
        'geometry': fin_gdf['geometry'].to_list()
        }
    )

    return fin_gdf


def get_mesonet_locations(
        geometry: Union[Path, gpd.GeoDataFrame]
        ) -> gpd.GeoDataFrame:
    '''
    Queries mesonet stations using polygon geometry
    :param geometry: Path or geopandas.GeoDataFrame - polygon geometry of area that stations will be extracted
    :return: geopandas.GeoDataFrame of station with metadata
    '''
    mesonet_stns = chmdata.mesonet.stns_metadata()
    stn_list = list(mesonet_stns.keys())
    dic_list = []
    for stn in stn_list:
        stn_dic = mesonet_stns[stn]
        dic = {}
        dic['properties'] = stn_dic
        dic['geometry'] = {'coordinates': [stn_dic['longitude'], stn_dic['latitude']], 'type': 'point'}

        dic_list.append(dic)

    gdf_return = gpd.GeoDataFrame.from_features(dic_list)
    gdf_return = gdf_return.drop(['longitude', 'latitude'], axis=1)
    fin_gdf = gdf_return

    if geometry is not None:
        if isinstance(geometry, (str, Path)):
            in_geom = gpd.read_file(geometry)
        elif isinstance(geometry, gpd.GeoDataFrame):
            in_geom = geometry
        else:
            raise ValueError("The input geometry is neither a string path nor a geopandas GeoDataFrame.")

        in_geom = in_geom.to_crs(4326)
        fin_gdf = fin_gdf.loc[fin_gdf.intersects(in_geom.geometry[0]), :]

    fin_gdf = gpd.GeoDataFrame(
        {'name': fin_gdf['name'].to_list(),
        'network': ['mesonet'] * len(fin_gdf),
        'sub_network': fin_gdf['sub_network'].to_list(),
        'install': pd.to_datetime(fin_gdf['date_installed']).to_list(),
        'geometry': fin_gdf['geometry'].to_list()
        }
    )

    return fin_gdf


def get_mesonet_data(ms_locs: gpd.GeoDataFrame,
                     variables: Union[list, None] = "",
                     start: str = "",
                     end: str = ""
                     ) -> xr.Dataset:
    '''
    Retreives data for all provided mesonet stations from the Montana Climate Office
    :param ms_locs: geopandas.GeoDataFrame - dataframe with station names where data will be retreived
    :param elems: list of str, desired variables from OBSERVATIONS to fetch. By default, all variables will be
    downloaded.
    :param start: str, optional; YYYY-MM-DD format (inclusive)
    :param end: str, optional; YYYY-MM-DD format (exclusive)
    :return: xr.Dataset with time and location dimentions with all data retreived
    '''
    hm_list = []
    ag_list = []
    for i in range(len(ms_locs)):
        mesonet_moc = chmdata.mesonet.Mesonet(stn_name=ms_locs['name'][i])
        data = mesonet_moc.get_data()

        xda_list = []
        for j in range(len(data.columns[1:])):
            xda = xr.DataArray(
                data=np.array([data[data.columns[1:][j]].values]).transpose(),
                dims=['time', 'location'],
                coords=dict(
                    time=(['time'], data.index),
                    location=(['location'], [list(ms_locs['name'])[i]])
                ),
                attrs=dict(
                    description=data.columns[1:][j][:data.columns[1:][j].index(' [')],
                    units=data.columns[1:][j][data.columns[1:][j].index("[") + 1:
                                              data.columns[1:][j].index("]")]
                ),
                name=data.columns[1:][j][:data.columns[1:][j].index(' [')]
            )
            xda_list.append(xda)
        xds = xr.merge(xda_list)

        if ms_locs['sub_network'][i] == 'HydroMet':
            hm_list.append(xds)

        if ms_locs['sub_network'][i] == 'AgriMet':
            ag_list.append(xds)

        hm_output = xr.merge(hm_list)
        ag_output = xr.merge(ag_list)

    return (hm_output, ag_output)


def get_agrimet_data(
        ag_locs: gpd.GeoDataFrame
) -> xr.Dataset:
    '''
    Retreives data for all provided agrimet stations from the U.S. Bureau of Reclamation
    :param ag_locs: geopandas.GeoDataFrame - dataframe with agrimet station names where data will be retreived
    :return: xr.Dataset with time and location dimentions with all data retreived
    '''

    # pull agrimet dataframe for locations in polygon
    loc_xds_list = []
    for i in range(len(ag_locs)):
        agrimet_moc = chmdata.agrimet.Agrimet(station=ag_locs['name'][i])
        data = agrimet_moc.fetch_met_data()

        # reformat dataframe to dataarray
        xda_list = []
        for j in range(len(data.columns)):
            xda = xr.DataArray(
                data=data[data.columns[j][0]].values,
                dims=['time', 'location'],
                coords=dict(
                    time=(['time'], data.index),
                    location=(['location'], [list(ag_locs['name'])[i]])
                    # [0] needs to be updated when looping through multipule sites
                ),
                attrs=dict(
                    description=data.columns[j][1],
                    units=data.columns[j][2][1:-1]
                ),
                name=data.columns[j][0]
            )
            # merge variable dataarrays for a location
            xda_list.append(xda)
        loc_xds = xr.merge(xda_list)

        # merge location datasets
        loc_xds_list.append(loc_xds)
    output = xr.merge(loc_xds_list)

    return output