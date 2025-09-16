import sys
import geopandas as gpd
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zeep import Client, helpers
from typing import Union
from datetime import datetime, timedelta

import dataretrieval.nwis as nwis
from chmdata import agrimet
from chmdata import mesonet

sys.path.append("C:/Users/CND905/Downloaded_Programs/MTDNRCdata")
import MTDNRCdata


DEFAULT_DATES = ('1900-01-01', (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d"))

def get_gauge_locations(geometry: Union[Path, gpd.GeoDataFrame], plot=False) -> gpd.GeoDataFrame:
    """
    Function to query USGS and DNRC stream gauge locations using polygon geometry
    :param geometry: Path or geopandas.GeoDataFrame - polygon geometry of area that stations will be extracted from in crs:4326
    :param plot: bool - whether to plot station locations or not; default is False
    return: gpd.GeoDataFrame of gauge locations within the polygon area
    """

    if geometry is not None:
        if isinstance(geometry, (str, Path)):
            in_geom = gpd.read_file(geometry)
        elif isinstance(geometry, gpd.GeoDataFrame):
            in_geom = geometry
        else:
            raise ValueError("The input geometry is neither a string path nor a geopandas GeoDataFrame.")
    if len(geometry) > 1:
        raise ValueError("Only one geometry can be entered.")

    # get STAGE sites within geometry, site type and status
    gdf = MTDNRCdata.stage.get_site_locations(geometry=in_geom)
    dnrc_gdf = gpd.GeoDataFrame(
        {'network': ['DNRC'] * len(gdf),
            'id': gdf['LocationCode'].to_list(),
            'name': gdf['LocationName'].to_list(),
            'type': gdf['StatusDesc'].to_list(),
            'geometry': gdf['geometry'].to_list()
            })

    # get USGS sites within bbox of geometry, parameter and status
    in_geom = in_geom.to_crs(4326)
    bnds = list(np.round(in_geom.bounds.values[0], decimals=6))
    gdf = nwis.what_sites(bBox=bnds)[0]
    usgs_gdf = gpd.GeoDataFrame(
        {'network': ['USGS'] * len(gdf),
            'id': gdf['site_no'].to_list(),
            'name': gdf['station_nm'].to_list(),
            'type': gdf['site_tp_cd'].to_list(),
            'geometry': gdf['geometry'].to_list()
            })


    gdf = pd.concat([dnrc_gdf, usgs_gdf]).reset_index(drop=True)

    # filter gdf to sites intersecting with geometry
    gdf = gdf.set_crs(4326)
    out = gdf.loc[gdf.intersects(in_geom.geometry[0]), :].reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots()
        in_geom.boundary.plot(ax=ax, color='black')
        gdf.plot(column='id', ax=ax, legend=True, cmap='Accent')

    return out


def get_gauge_parameters(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Function to add available parameters at DNRC and USGS gauge locations
    :param gdf: geopandas.GeoDataFrame - gauge location information with parameters columns
    return: gpd.GeoDataFrame of gauge locations information with available parameters
    """
    param_list = []
    for i in range(len(gdf)):
        site_id = gdf.iloc[i]['id']
        if gdf.iloc[i]['network'] == "DNRC":
            data = MTDNRCdata.stage.get_location_parameters(site_ids=site_id)
            params = data['Parameter'].values.tolist()
            param_list.append(list(set(params)))

        if gdf.iloc[i]['network'] == "USGS":
            data = nwis.get_record(sites=site_id)
            params = [p[:5] for p in list(data)[1:]]
            param_list.append(list(set(params)))

    gdf['parameters'] = param_list

    return gdf


def get_stage_gauge_data(sites: Union[list, gpd.GeoDataFrame], dataset: str, timestep: str, start=DEFAULT_DATES[0], end=DEFAULT_DATES[1]) -> xr.Dataset:
    """
    Retrieve data from multiple DNRC stream gauges with xarray implementation
    :param sites: gpd.GeoDataFrame or list - dnrc gauge sites with 'id' column of location codes to be retrieved
    :param dataset: str - data variable retrieved; specify 'QR' for discharge, 'HG' for stage, or 'TW' for water temperature
    :param timestep: str - frequency of data retrieved; specify either 'instant' for instantaneous data or 'daily' for
     average daily values
    :param start: str - start date of period of interest, in YYYY-MM-DD format; defult is 1900-01-01
    :param end: str - end date of period of interest, in YYYY-MM-DD format; defult is the prior days date
    return: xr.Dataset of data requested from gauge locations
    """
    if sites is not None:
        if isinstance(sites, gpd.GeoDataFrame):
            in_sites = list(sites['id'])
        elif isinstance(sites, list):
            in_sites = sites
        else:
            raise ValueError("The sites input is neither a geopandas GeoDataFrame nor a list.")

    xda_list = []
    for i in range(len(in_sites)):
        site_id = in_sites[i]

        data = MTDNRCdata.stage.GetSite(site_id, timestep=timestep, dataset=dataset, start=start, end=end)
        data = data.data

        xda = xr.DataArray(
            data=np.array([data['RecordedValue'].values]).transpose(),
            dims=['time', 'location'],
            coords=dict(
                time=(['time'], np.array(pd.to_datetime(data.index, utc=True))),
                location=(['location'], [data.iloc[0]['SiteID']])
            ),
            attrs=dict(
                description=data['DatasetLabel'].values[0][:data['DatasetLabel'].values[0].rfind('(')],
                units=data['DatasetLabel'].values[0][data['DatasetLabel'].values[0].rfind('_') + 1:]
            ),
            name=dataset
        )

        xda_list.append(xda)
    output = xr.merge(xda_list)

    return output


def get_usgs_gauge_data(sites: Union[list, gpd.GeoDataFrame], dataset: str, timestep: str, start=DEFAULT_DATES[0],
                        end=DEFAULT_DATES[1]) -> xr.Dataset:
    """
    Retrieve data from multiple USGS stream gauges with xarray implementation
    :param sites: gpd.GeoDataFrame or list - usgs gauge sites with 'id' column of location codes to be retrieved
    :param dataset: str - data variable retrieved
    :param timestep: str - frequency of data retrieved; specify either 'iv' for instantaneous data or 'dv' for
     average daily values
    :param start: str - start date of period of interest, in YYYY-MM-DD format; defult is 1900-01-01
    :param end: str - end date of period of interest, in YYYY-MM-DD format; defult is the prior days date
    return: xr.Dataset of data requested from gauge locations
    """

    if sites is not None:
        if isinstance(sites, gpd.GeoDataFrame):
            in_sites = list(sites['id'])
        elif isinstance(sites, list):
            in_sites = sites
        else:
            raise ValueError("The sites input is neither a geopandas GeoDataFrame nor a list.")

    xda_list = []
    for i in range(len(in_sites)):
        site_id = in_sites[i]

        data = nwis.get_record(sites=site_id, service=timestep, start=start, end=end, parameterCd=dataset)

        if len(data) == 0:
            continue

        xda = xr.DataArray(
            data=np.array(np.array([data.iloc[:, 1]])).transpose(),
            dims=['time', 'location'],
            coords=dict(
                time=(['time'], np.array(pd.to_datetime(data.index, utc=True))),
                location=(['location'], [data.iloc[0]['site_no']])),
            name=data.columns[1]
        )

        xda_list.append(xda)
    output = xr.merge(xda_list)

    return output


def get_station_locations(geometry: Union[Path, gpd.GeoDataFrame],
                          buffer_distance: int = 0,
                          network: str = 'both',
                          plot: bool = True) -> gpd.GeoDataFrame:
    """
    Queries mesonet and/or agrimet stations using a polygon geometry
    :param geometry: Path or geopandas.GeoDataFrame - polygon geometry of area that stations will be retrieved
    :param buffer_distance: int - distance in meters from geometry boundary that stations will be filtered; default is 0
    :param network: str - network that stations will be retrieved; specify 'agrimet', 'mesonet', or 'both'; default is both
    :param plot: bool - whether to plot station locations or not; default is True
    :return: geopandas.GeoDataFrame of stations and metadata within polygon and buffer area
    """
    if geometry is not None:
        if isinstance(geometry, (str, Path)):
            in_geom = gpd.read_file(geometry)
        elif isinstance(geometry, gpd.GeoDataFrame):
            in_geom = geometry
        else:
            raise ValueError("The input geometry is neither a string path nor a geopandas GeoDataFrame.")

    # get all agrimet stations
    if network == 'agrimet' or network == 'both':
        agrimet_stns = agrimet.load_stations()
        stn_list = list(agrimet_stns.keys())
        dic_list = []
        for stn in stn_list:
            dic_list.append(agrimet_stns[stn])
        gdf = gpd.GeoDataFrame.from_features(dic_list)

        gdf = gpd.GeoDataFrame(
            {'name': gdf['siteid'].to_list(),
             'network': ['agrimet'] * len(gdf),
             'sub_network': gdf['type'].to_list(),
             'install': gdf['install'].to_list(),
             'geometry': gdf['geometry'].to_list()
             }
        )
        # rename gdf so it is not overwritten before concatenating
        if network == 'both':
            a_gdf = gdf

    # get all mesonet stations
    if network == 'mesonet' or network == 'both':
        mesonet_stns = mesonet.stns_metadata()
        stn_list = list(mesonet_stns.keys())
        dic_list = []
        for stn in stn_list:
            stn_dic = mesonet_stns[stn]
            dic = {
                'properties': stn_dic,
                'geometry': {'coordinates': [stn_dic['longitude'], stn_dic['latitude']], 'type': 'point'}
            }
            dic_list.append(dic)
        gdf = gpd.GeoDataFrame.from_features(dic_list)

        gdf = gpd.GeoDataFrame(
            {'name': gdf['name'].to_list(),
             'network': ['mesonet'] * len(gdf),
             'sub_network': gdf['sub_network'].to_list(),
             'install': gdf['date_installed'].to_list(),
             'geometry': gdf['geometry'].to_list()
             }
        )
        # rename gdf so it is not overwritten before concatenating
        if network == 'both':
            m_gdf = gdf

    # concatenate gdf from both networks
    if network == 'both':
        gdf = pd.concat([a_gdf, m_gdf]).reset_index(drop=True)

    # filter for stations within geometry buffer area
    gdf = gdf.set_crs(4326).to_crs(5071)
    in_geom = in_geom.to_crs(5071)
    poly = in_geom.geometry[0]
    poly = poly.buffer(distance=buffer_distance)
    gdf = gdf.loc[gdf.intersects(poly), :]
    out = gdf.to_crs(4326)

    if plot:
        fig, ax = plt.subplots()
        in_geom.boundary.plot(ax=ax, color='black')
        gdf.plot(column='network', ax=ax, legend=True, cmap='Accent')

    return out


def get_mesonet_station_data(sites: Union[gpd.GeoDataFrame, list],
                             dataset: str = '',
                             start: str = '',
                             end: str = '') -> xr.Dataset:
    """
    Retrieves data for all provided mesonet stations from the Montana Climate Office
    :param sites: gpd.GeoDataFrame or list - station names where data will be retrieved; gpd.GeoDataFrame column
    must be titled 'name'
    :param dataset: list, desired variables to retrieve as strings; default will retrieve all available variables
    :param start: str, optional; YYYY-MM-DD format; default will retrieve all available dates
    :param end: str, optional; YYYY-MM-DD format; default will retrieve all available dates
    :return: two xr.Datasets of requested datasets from sites and time period; Hydromet and Agrimet xr.Datasets are
     indexable respectively
    """
    if sites is not None:
        if isinstance(sites, gpd.GeoDataFrame):
            in_sites = list(sites['name'].values)
        if isinstance(sites, list):
            in_sites = sites
    else:
        raise ValueError('Must provide a sites input as a geopandas GeoDataFrame or list.')

    hm_list = []
    ag_list = []
    met_list = []
    for i in range(len(in_sites)):
        meso_site = mesonet.Mesonet(stn_name=in_sites[i])
        data = meso_site.get_data(elems=dataset, start=start, end=end)
        met_list.append(meso_site.all_metadata[meso_site.find_stn_abr(in_sites[0])]['sub_network'])

        xda_list = []
        for j in range(len(data.columns[1:])):
            xda = xr.DataArray(
                data=np.array([data[data.columns[1:][j]].values]).transpose(),
                dims=['time', 'location'],
                coords=dict(
                    time=(['time'], data.index),
                    location=(['location'], [list(sites['name'])[i]])),
                attrs=dict(
                    description=data.columns[1:][j][:data.columns[1:][j].index(' [')],
                    units=data.columns[1:][j][data.columns[1:][j].index("[") + 1: data.columns[1:][j].index("]")]),
                name=data.columns[1:][j][:data.columns[1:][j].index(' [')]
            )
            xda_list.append(xda)
        xds = xr.merge(xda_list)

        if met_list[i] == 'HydroMet':
            hm_list.append(xds)
        hm_output = xr.merge(hm_list)
        hm_output.attrs.clear()

        if met_list[i] == 'AgriMet':
            ag_list.append(xds)
        ag_output = xr.merge(ag_list)
        ag_output.attrs.clear()

    return hm_output, ag_output


def get_agrimet_station_data(sites: Union[list, gpd.GeoDataFrame],
                             dataset: str = '',
                             start: str = '',
                             end: str = '') -> xr.Dataset:
    """
    Retrieves data for all provided agrimet stations from the US Bureau of Reclamation regional websites'
    :param sites: gpd.GeoDataFrame or list - station names where data will be retrieved; gpd.GeoDataFrame column must
     be titled 'name'
    :param dataset: list, desired variables to retrieve as strings; default will retrieve all available variables
    :param start: str, optional; YYYY-MM-DD format; default will retrieve all available dates
    :param end: str, optional; YYYY-MM-DD format; default will retrieve all available dates
    :return: xr.Datasets of requested datasets from sites and time period
    """
    if sites is not None:
        if isinstance(sites, gpd.GeoDataFrame):
            in_sites = list(sites['name'].values)
        if isinstance(sites, list):
            in_sites = sites

    if start == '' and end == '':
        start = None
        end = None

    # pull agrimet data for each site
    xds_list = []
    for i in range(len(in_sites)):
        # filter by date and variables
        agri_site = agrimet.Agrimet(station=in_sites[i], start_date=start, end_date=end)
        data = agri_site.fetch_met_data()
        if dataset == '':
            pass
        else:
            data = data[dataset]

        # reformat dataframe to dataarray
        xda_list = []
        for j in range(len(data.columns)):
            xda = xr.DataArray(
                data=data[data.columns[j][0]].values,
                dims=['time', 'location'],
                coords=dict(
                    time=(['time'], data.index),
                    location=(['location'], [in_sites[i]])),
                attrs=dict(
                    description=data.columns[j][1],
                    units=data.columns[j][2][1:-1]),
                name=data.columns[j][0])
            # merge variable dataarrays for a location
            xda_list.append(xda)
        loc_xds = xr.merge(xda_list)

        # merge site dataarrays
        xds_list.append(loc_xds)
    output = xr.merge(xds_list)
    output.attrs.clear()

    return output



# should go in a config file
NRCS = Client('https://wcc.sc.egov.usda.gov/awdbWebService/services?WSDL')

DATASET_DICT = {
    'elementCd': ['WTEQ', 'PREC', 'TMAX', 'TMIN'],
    'description': ['snow water equivalent', 'accumulated precipitation', 'temperature maximum', 'temperature minimum'],
    'units': ['inches', 'inches', 'degree F', 'degree F']
    }


def get_nrcs_station_locations(geometry: Union[Path, gpd.GeoDataFrame], buffer_distance: int = 0, network: list = ['SNTL'], plot: bool = True) -> gpd.GeoDataFrame:
    """
    Queries US Natural Resources Conservation Services snotel stations using a polygon geometry
    :param geometry: Path or geopandas.GeoDataFrame - polygon geometry of area that stations will be retrieved
    :param buffer_distance: int - distance in meters from geometry boundary that stations will be filtered; defult is 0
    :param network: list - network that stations will be retrieved; default is 'SNTL'
    :param plot: bool - whether to plot station locations or not; default is True
    :return: geopandas.GeoDataFrame of stations and metadata within polygon and buffer area
    """
    if geometry is not None:
        if isinstance(geometry, (str, Path)):
            in_geom = gpd.read_file(geometry)
        elif isinstance(geometry, gpd.GeoDataFrame):
            in_geom = geometry
        else:
            raise ValueError("The input geometry is neither a string path nor a geopandas GeoDataFrame.")

    stns = NRCS.service.getStations(stateCds = ["MT", "ID", "WY", "ND", "SD"], networkCds = network, logicalAnd = True)
    meta_stns = NRCS.service.getStationMetadataMultiple(stns)
    df = pd.DataFrame(helpers.serialize_object(meta_stns))
    geom = gpd.points_from_xy(df['longitude'], df['latitude'], crs=4326)
    gdf = gpd.GeoDataFrame(df, geometry=geom)

    # buffer and search gdf
    gdf = gdf.set_crs(4326).to_crs(5071)
    tran_geom = in_geom.to_crs(5071)
    poly = tran_geom.geometry[0]
    poly = poly.buffer(distance=buffer_distance)
    gdf = gdf.loc[gdf.intersects(poly), :]
    out = gdf.to_crs(4326)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 15))
        in_geom.boundary.plot(ax=ax, color='black')
        out.plot(column='stationTriplet', ax=ax, legend=True, cmap='Accent')
        leg = ax.get_legend()
        leg.set_bbox_to_anchor((1.4, 0.4)) # Example: right center outside the plot
        plt.show()

    out.reset_index()

    return out


def get_snotel_station_data(sites: Union[list, gpd.GeoDataFrame],
                            dataset: list,
                            start: str,
                            end: str,
                            timestep: str = 'DAILY') -> xr.Dataset:
    """
    Retrieves data for all provided snotel stations from the US Natural Resource Conservation Service API
    :param sites: gpd.GeoDataFrame or list - station names where data will be retrieved; gpd.GeoDataFrame column must be titled 'name'
    :param dataset: list, desired variables to retrieve as strings; default will retrieve all available variables
    :param start: str, YYYY-MM-DD format; default will retrieve all available dates
    :param end: str, YYYY-MM-DD format; default will retrieve all available dates
    :param timestep: str, frequency of data retrieved; default is 'DAILY'
    :return: xr.Datasets of requested datasets from sites and time period
    """

    def key_to_col(d_list, d_key):
        v_list = []
        for i in d_list:
            if d_key in i:
                v_list.append(i[d_key])
        return v_list

    if sites is not None:
        if isinstance(sites, gpd.GeoDataFrame):
            in_sites = list(sites['name'].values)
        if isinstance(sites, list):
            in_sites = sites

    dataset_list = []
    for element in dataset:
        dat = NRCS.service.getData(stationTriplets=in_sites,
                                   elementCd=element,
                                   ordinal=1,
                                   getFlags='true',
                                   duration=timestep,
                                   beginDate=start,
                                   endDate=end,
                                   alwaysReturnDailyFeb29=False)

        dat_dict = helpers.serialize_object(dat)
        var = np.array(key_to_col(dat_dict, d_key='values')).astype(float)
        flg = np.array(key_to_col(dat_dict, d_key='flags'))

        element_list = []
        for dat in list([var, flg]):

            if isinstance(dat[0][0], float):
                name = element
                description = DATASET_DICT['description'][DATASET_DICT['elementCd'].index(element)]
                units = DATASET_DICT['units'][DATASET_DICT['elementCd'].index(element)]
            if isinstance(dat[0][0], str):
                name = element + '_flag'
                description = element + '_flag'
                units = 'flag'

            xda = xr.DataArray(
                data=dat.transpose(),
                dims=['time', 'location'],
                coords=dict(
                    time=(['time'], pd.date_range(start=dat_dict[0]['beginDate'], end=dat_dict[0]['endDate'])),
                    location=(['location'], in_sites)),
                attrs=dict(
                    description=description,
                    units=units),
                name=name
            )

            element_list.append(xda)
        element_xds = xr.merge(element_list)

        dataset_list.append(element_xds)
    output = xr.merge(dataset_list)

    return output