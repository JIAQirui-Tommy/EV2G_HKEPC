'''
This file contains the loaders for the EV City environment.
'''

import numpy as np
import pandas as pd
import math
import datetime
import pkg_resources
import json
import os
from typing import List, Tuple

from ev2gym.models.ev_charger import EV_Charger
from ev2gym.models.ev import EV
from ev2gym.models.transformer import Transformer
from ev2gym.models.grid import PowerGrid

from ev2gym.utilities.utils import EV_spawner, generate_power_setpoints, EV_spawner_GF


def apply_workplace_dual_peak_profile(env) -> None:
    """Add an evening commute peak to the workplace scenario.

    The original workplace data is daytime-heavy. For simulations that start
    later in the day, we mirror part of the morning profile into the evening so
    the environment still produces a meaningful EV population.
    """

    cfg = env.config.get('workplace_dual_peak', {})
    if not cfg.get('enabled', False):
        return

    evening_start_hour = cfg.get('evening_start_hour', 18)
    evening_end_hour = cfg.get('evening_end_hour', 23)
    source_shift_hours = cfg.get('source_shift_hours', 12)
    arrival_scale = cfg.get('arrival_scale', 1.0)
    min_stay_hours = cfg.get('min_stay_hours', 6.0)
    min_energy_kwh = cfg.get('min_energy_kwh', 14.0)

    if 'workplace' not in env.df_arrival_week.columns:
        return

    start_idx_15 = max(0, evening_start_hour * 4)
    end_idx_15 = min(len(env.df_arrival_week), evening_end_hour * 4 + 4)
    shift_idx_15 = source_shift_hours * 4

    for idx in range(start_idx_15, end_idx_15):
        src_idx = idx - shift_idx_15
        if 0 <= src_idx < len(env.df_arrival_week):
            mirrored_value = env.df_arrival_week.at[src_idx, 'workplace'] * arrival_scale
            env.df_arrival_week.at[idx, 'workplace'] = max(
                env.df_arrival_week.at[idx, 'workplace'],
                mirrored_value,
            )

    start_idx_30 = max(0, evening_start_hour * 2)
    end_idx_30 = min(len(env.df_req_energy), evening_end_hour * 2 + 2)
    shift_idx_30 = source_shift_hours * 2

    for idx in range(start_idx_30, end_idx_30):
        src_idx = idx - shift_idx_30
        if 0 <= src_idx < len(env.df_req_energy):
            src_energy = env.df_req_energy.at[src_idx, 'workplace']
            src_stay = env.df_time_of_stay_vs_arrival.at[src_idx, 'workplace']

            if pd.isna(src_energy):
                src_energy = min_energy_kwh
            if pd.isna(src_stay):
                src_stay = min_stay_hours

            env.df_req_energy.at[idx, 'workplace'] = max(src_energy, min_energy_kwh)
            env.df_time_of_stay_vs_arrival.at[idx, 'workplace'] = max(src_stay, min_stay_hours)


def load_ev_spawn_scenarios(env) -> None:
    '''Loads the EV spawn scenarios of the simulation'''

    # Load the EV specs
    if env.config['heterogeneous_ev_specs']:

        if "ev_specs_file" in env.config:
            ev_specs_file = env.config['ev_specs_file']
        else:
            ev_specs_file = pkg_resources.resource_filename(
                'ev2gym', 'data/ev_specs.json')

        with open(ev_specs_file) as f:
            env.ev_specs = json.load(f)

        registrations = np.zeros(len(env.ev_specs.keys()))
        for i, ev_name in enumerate(env.ev_specs.keys()):
            # sum the total number of registrations
            registrations[i] = env.ev_specs[ev_name]['number_of_registrations']

        env.normalized_ev_registrations = registrations/registrations.sum()

    if env.scenario == 'GF':
        env.df_arrival = np.load('./GF_data/time_of_arrival.npy')  # weekdays
        env.time_of_connection_vs_hour_weekday = np.load(
            './GF_data/weekday_time_of_stay.npy')
        env.time_of_connection_vs_hour_weekend = np.load(
            './GF_data/weekend_time_of_stay.npy')
        env.df_req_energy_weekday = np.load('./GF_data/weekday_volumeKWh.npy')
        env.df_req_energy_weekend = np.load('./GF_data/weekend_volumeKWh.npy')

        return

    df_arrival_week_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-arrival.csv')
    df_arrival_weekend_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-arrival-weekend.csv')
    df_connection_time_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-connection-time.csv')
    df_energy_demand_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-energy-demand.csv')
    time_of_connection_vs_hour_file = pkg_resources.resource_filename(
        'ev2gym', 'data/time_of_connection_vs_hour.npy')

    df_req_energy_file = pkg_resources.resource_filename(
        'ev2gym', 'data/mean-demand-per-arrival.csv')
    df_time_of_stay_vs_arrival_file = pkg_resources.resource_filename(
        'ev2gym', 'data/mean-session-length-per.csv')

    env.df_arrival_week = pd.read_csv(df_arrival_week_file)  # weekdays
    env.df_arrival_weekend = pd.read_csv(df_arrival_weekend_file)  # weekends
    env.df_connection_time = pd.read_csv(
        df_connection_time_file)  # connection time
    env.df_energy_demand = pd.read_csv(df_energy_demand_file)  # energy demand
    env.time_of_connection_vs_hour = np.load(
        time_of_connection_vs_hour_file)  # time of connection vs hour

    env.df_req_energy = pd.read_csv(
        df_req_energy_file)  # energy demand per arrival
    # replace column work with workplace
    env.df_req_energy = env.df_req_energy.rename(columns={'work': 'workplace',
                                                          'home': 'private'})
    env.df_req_energy = env.df_req_energy.fillna(0)

    env.df_time_of_stay_vs_arrival = pd.read_csv(
        df_time_of_stay_vs_arrival_file)  # time of stay vs arrival
    env.df_time_of_stay_vs_arrival = env.df_time_of_stay_vs_arrival.fillna(0)
    env.df_time_of_stay_vs_arrival = env.df_time_of_stay_vs_arrival.rename(columns={'work': 'workplace',
                                                                                    'home': 'private'})
    apply_workplace_dual_peak_profile(env)


def load_power_setpoints(env) -> np.ndarray:
    '''
    Loads the power setpoints of the simulation based on the day-ahead prices
    '''

    if env.load_from_replay_path:
        return env.replay.power_setpoints
    else:
        if not env.config['power_setpoint_enabled']:
            return np.zeros(env.simulation_length)
        else:
            return generate_power_setpoints(env)


def generate_residential_inflexible_loads(env) -> np.ndarray:
    '''
    This function loads the inflexible loads of each transformer
    in the simulation.
    '''

    # Load the data
    data_path = pkg_resources.resource_filename(
        'ev2gym', 'data/residential_loads.csv')
    data = pd.read_csv(data_path, header=None)

    desired_timescale = env.timescale
    simulation_length = env.simulation_length
    simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')
    number_of_transformers = env.number_of_transformers

    dataset_timescale = 15
    dataset_starting_date = '2022-01-01 00:00:00'

    if desired_timescale > dataset_timescale:
        data = data.groupby(
            data.index // (desired_timescale/dataset_timescale)).max()
    elif desired_timescale < dataset_timescale:
        # extend the dataset to data.shape[0] * (dataset_timescale/desired_timescale)
        # by repeating the data every (dataset_timescale/desired_timescale) rows
        data = data.loc[data.index.repeat(
            dataset_timescale/desired_timescale)].reset_index(drop=True)

    # duplicate the data to have two years of data
    data = pd.concat([data, data], ignore_index=True)

    # add a date column to the dataframe
    data['date'] = pd.date_range(
        start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')

    # find year of the data
    year = int(dataset_starting_date.split('-')[0])
    # replace the year of the simulation date with the year of the data
    simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'

    simulation_index = data[data['date'] == simulation_date].index[0]

    # select the data for the simulation date
    data = data[simulation_index:simulation_index+simulation_length]

    # drop the date column
    data = data.drop(columns=['date'])
    new_data = pd.DataFrame()

    for i in range(number_of_transformers):
        new_data['tr_'+str(i)] = data.sample(10, axis=1,
                                             random_state=env.tr_seed).sum(axis=1)

    # return the "tr_" columns
    return new_data.to_numpy().T


def load_prophet_inflexible_loads(env) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load external load forecast data, preferably Prophet hourly output, and
    align it to the simulation horizon.

    Returns:
        - actual_loads: matrix (n_transformers, simulation_length)
        - forecast_loads: matrix (n_transformers, simulation_length)
    '''
    cfg = env.config.get('load_forecast', {})
    file_path = cfg.get('file')
    if not file_path:
        raise FileNotFoundError('Missing load_forecast.file in config')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Load forecast file not found: {file_path}')

    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path)

    datetime_column = cfg.get('datetime_column', 'ds')
    value_column = cfg.get('value_column', 'yhat')
    actual_value_column = cfg.get('actual_value_column')

    if datetime_column not in data.columns:
        raise KeyError(f'Missing datetime column "{datetime_column}" in {file_path}')
    if value_column not in data.columns:
        raise KeyError(f'Missing forecast value column "{value_column}" in {file_path}')

    data = data.copy()
    data[datetime_column] = pd.to_datetime(data[datetime_column])
    data = data.sort_values(datetime_column).drop_duplicates(datetime_column)
    data = data.set_index(datetime_column)

    desired_index = pd.date_range(
        start=env.sim_starting_date,
        periods=env.simulation_length,
        freq=f'{env.timescale}min',
    )

    aligned = data.reindex(data.index.union(desired_index)).sort_index().interpolate(method='time')
    aligned = aligned.reindex(desired_index)

    forecast_values = pd.to_numeric(aligned[value_column], errors='coerce').interpolate(
        method='linear', limit_direction='both'
    ).bfill().ffill()

    if actual_value_column and actual_value_column in aligned.columns:
        actual_values = pd.to_numeric(aligned[actual_value_column], errors='coerce').interpolate(
            method='linear', limit_direction='both'
        ).bfill().ffill()
    else:
        actual_values = forecast_values.copy()

    if forecast_values.isna().any():
        raw_forecast = pd.to_numeric(data[value_column], errors='coerce').dropna()
        fallback_forecast = float(raw_forecast.iloc[0]) if len(raw_forecast) > 0 else 0.0
        forecast_values = forecast_values.fillna(fallback_forecast)

    if actual_values.isna().any():
        raw_actual = pd.to_numeric(data[actual_value_column], errors='coerce').dropna() \
            if actual_value_column and actual_value_column in data.columns else pd.Series(dtype=float)
        if len(raw_actual) > 0:
            fallback_actual = float(raw_actual.iloc[0])
        else:
            fallback_actual = float(forecast_values.iloc[0]) if len(forecast_values) > 0 else 0.0
        actual_values = actual_values.fillna(fallback_actual)

    actual_series = np.maximum(actual_values.to_numpy(dtype=float), 0.0)
    forecast_series = np.maximum(forecast_values.to_numpy(dtype=float), 0.0)

    # Prophet input may come from a system-level load file (for example GWh for a
    # whole region). In that case we keep the temporal shape but rescale it to a
    # local transformer-level kW range so it is comparable with EV charging power.
    target_peak_kw = cfg.get('target_peak_kw')
    target_mean_kw = cfg.get('target_mean_kw')
    if target_peak_kw is not None:
        combined_peak = max(float(np.max(actual_series)), float(np.max(forecast_series)), 1e-6)
        scale = float(target_peak_kw) / combined_peak
        actual_series = actual_series * scale
        forecast_series = forecast_series * scale
    elif target_mean_kw is not None:
        combined_mean = max(float(np.mean(forecast_series)), 1e-6)
        scale = float(target_mean_kw) / combined_mean
        actual_series = actual_series * scale
        forecast_series = forecast_series * scale

    actual_loads = np.tile(actual_series, (env.number_of_transformers, 1))
    forecast_loads = np.tile(forecast_series, (env.number_of_transformers, 1))
    return actual_loads, forecast_loads


def generate_pv_generation(env) -> np.ndarray:
    '''
    This function loads the PV generation of each transformer by loading the data from a file
    and then adding minor variations to the data
    '''

    # Load the data
    data_path = pkg_resources.resource_filename(
        'ev2gym', 'data/pv_netherlands.csv')
    data = pd.read_csv(data_path, sep=',', header=0)
    data.drop(['time', 'local_time'], inplace=True, axis=1)

    desired_timescale = env.timescale
    simulation_length = env.simulation_length
    simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')
    number_of_transformers = env.number_of_transformers

    dataset_timescale = 60
    dataset_starting_date = '2019-01-01 00:00:00'

    if desired_timescale > dataset_timescale:
        data = data.groupby(
            data.index // (desired_timescale/dataset_timescale)).max()
    elif desired_timescale < dataset_timescale:
        # extend the dataset to data.shape[0] * (dataset_timescale/desired_timescale)
        # by repeating the data every (dataset_timescale/desired_timescale) rows
        data = data.loc[data.index.repeat(
            dataset_timescale/desired_timescale)].reset_index(drop=True)
        # data = data/ (dataset_timescale/desired_timescale)

    # smooth data by taking the mean of every 5 rows
    data['electricity'] = data['electricity'].rolling(
        window=60//desired_timescale, min_periods=1).mean()
    # use other type of smoothing
    data['electricity'] = data['electricity'].ewm(
        span=60//desired_timescale, adjust=True).mean()

    # duplicate the data to have two years of data
    data = pd.concat([data, data], ignore_index=True)

    # add a date column to the dataframe
    data['date'] = pd.date_range(
        start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')

    # find year of the data
    year = int(dataset_starting_date.split('-')[0])
    # replace the year of the simulation date with the year of the data
    simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'

    simulation_index = data[data['date'] == simulation_date].index[0]

    # select the data for the simulation date
    data = data[simulation_index:simulation_index+simulation_length]

    # drop the date column
    data = data.drop(columns=['date'])
    new_data = pd.DataFrame()

    for i in range(number_of_transformers):
        new_data['tr_'+str(i)] = data * env.tr_rng.uniform(0.9, 1.1)

    return new_data.to_numpy().T


def load_transformers(env) -> List[Transformer]:
    '''Loads the transformers of the simulation
    If load_from_replay_path is None, then the transformers are created randomly

    Returns:
        - transformers: a list of transformer objects
    '''

    if env.load_from_replay_path is not None:
        return env.replay.transformers

    transformers = []

    load_forecast_cfg = env.config.get('load_forecast', {})
    use_external_load_forecast = load_forecast_cfg.get('enabled', False)
    external_actual_loads = None
    external_forecast_loads = None

    if use_external_load_forecast:
        try:
            external_actual_loads, external_forecast_loads = load_prophet_inflexible_loads(env)
        except Exception as exc:
            print(f'Warning: failed to load external load forecast, falling back to default loads: {exc}')
            use_external_load_forecast = False

    if use_external_load_forecast:
        inflexible_loads = external_actual_loads
    elif env.config['inflexible_loads']['include']:

        if env.scenario == 'private':
            inflexible_loads = generate_residential_inflexible_loads(env)

        # TODO add inflexible loads for public and workplace scenarios
        else:
            inflexible_loads = generate_residential_inflexible_loads(env)

    else:
        inflexible_loads = np.zeros((env.number_of_transformers,
                                    env.simulation_length))

    if env.config['solar_power']['include']:
        solar_power = generate_pv_generation(env)
    else:
        solar_power = np.zeros((env.number_of_transformers,
                                env.simulation_length))

    if env.charging_network_topology:
        # parse the topology file and create the transformers
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            cs_ids = []
            for cs in env.charging_network_topology[tr]['charging_stations']:
                cs_ids.append(cs_counter)
                cs_counter += 1
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=cs_ids,
                                      max_power=env.charging_network_topology[tr]['max_power'],
                                      inflexible_load=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            if use_external_load_forecast:
                transformer.inflexible_load_forecast = np.clip(
                    external_forecast_loads[i, :],
                    transformer.min_power,
                    transformer.max_power,
                )
            transformers.append(transformer)

    else:
        # if env.number_of_transformers > env.cs:
        #     raise ValueError(
        #         'The number of transformers cannot be greater than the number of charging stations')
        for i in range(env.number_of_transformers):
            # get indexes where the transformer is connected
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=np.where(
                                          np.array(env.cs_transformers) == i)[0],
                                      max_power=env.config['transformer']['max_power'],
                                      inflexible_load=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            if use_external_load_forecast:
                transformer.inflexible_load_forecast = np.clip(
                    external_forecast_loads[i, :],
                    transformer.min_power,
                    transformer.max_power,
                )
            transformers.append(transformer)
    env.n_transformers = len(transformers)
    return transformers


def load_ev_charger_profiles(env) -> List[EV_Charger]:
    '''Loads the EV charger profiles of the simulation
    If load_from_replay_path is None, then the EV charger profiles are created randomly

    Returns:
        - ev_charger_profiles: a list of ev_charger_profile objects'''

    charging_stations = []
    if env.load_from_replay_path is not None:
        return env.replay.charging_stations

    v2g_enabled = env.config['v2g_enabled']
    wireless_config = env.config.get('wireless_charging', {})
    wireless_enabled = wireless_config.get('enabled', False)

    if wireless_enabled:
        zone_count = wireless_config.get('number_of_zones', env.cs)
        zone_n_ports = wireless_config.get('zone_n_ports', env.number_of_ports_per_cs)
        max_charge_current = wireless_config.get(
            'zone_max_charge_current',
            env.config['charging_station']['max_charge_current'],
        )
        min_charge_current = wireless_config.get(
            'zone_min_charge_current',
            env.config['charging_station']['min_charge_current'],
        )
        if v2g_enabled:
            max_discharge_current = wireless_config.get(
                'zone_max_discharge_current',
                env.config['charging_station']['max_discharge_current'],
            )
            min_discharge_current = wireless_config.get(
                'zone_min_discharge_current',
                env.config['charging_station']['min_discharge_current'],
            )
        else:
            max_discharge_current = 0
            min_discharge_current = 0

        voltage = wireless_config.get(
            'zone_voltage',
            env.config['charging_station']['voltage'],
        )
        phases = wireless_config.get(
            'zone_phases',
            env.config['charging_station']['phases'],
        )

        charging_stations = []
        env.cs = zone_count
        for zone_id in range(zone_count):
            transformer_id = env.cs_transformers[zone_id]
            ev_charger = EV_Charger(
                id=zone_id,
                connected_bus=transformer_id,
                connected_transformer=transformer_id,
                n_ports=zone_n_ports,
                max_charge_current=max_charge_current,
                min_charge_current=min_charge_current,
                max_discharge_current=max_discharge_current,
                min_discharge_current=min_discharge_current,
                phases=phases,
                voltage=voltage,
                timescale=env.timescale,
                verbose=env.verbose,
                is_wireless_zone=True,
            )
            charging_stations.append(ev_charger)
        return charging_stations

    if env.charging_network_topology:
        # parse the topology file and create the charging stations
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            for cs in env.charging_network_topology[tr]['charging_stations']:
                ev_charger = EV_Charger(id=cs_counter,
                                        connected_bus=i,
                                        connected_transformer=i,
                                        min_charge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['min_charge_current'],
                                        max_charge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['max_charge_current'],
                                        min_discharge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['min_discharge_current'],
                                        max_discharge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['max_discharge_current'],
                                        voltage=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['voltage'],
                                        n_ports=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['n_ports'],
                                        charger_type=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['charger_type'],
                                        phases=env.charging_network_topology[tr]['charging_stations'][cs]['phases'],
                                        timescale=env.timescale,
                                        verbose=env.verbose,)
                cs_counter += 1
                charging_stations.append(ev_charger)
        env.cs = len(charging_stations)
        return charging_stations

    else:
        if v2g_enabled:
            max_discharge_current = env.config['charging_station']['max_discharge_current']
            min_discharge_current = env.config['charging_station']['min_discharge_current']
        else:
            max_discharge_current = 0
            min_discharge_current = 0

        for i in range(env.cs):
            ev_charger = EV_Charger(id=i,
                                    connected_bus=env.cs_transformers[i],
                                    connected_transformer=env.cs_transformers[i],
                                    n_ports=env.number_of_ports_per_cs,
                                    max_charge_current=env.config['charging_station']['max_charge_current'],
                                    min_charge_current=env.config['charging_station']['min_charge_current'],
                                    max_discharge_current=max_discharge_current,
                                    min_discharge_current=min_discharge_current,
                                    phases=env.config['charging_station']['phases'],
                                    voltage=env.config['charging_station']['voltage'],
                                    timescale=env.timescale,
                                    verbose=env.verbose,)

            charging_stations.append(ev_charger)
        return charging_stations


def load_ev_profiles(env) -> List[EV]:
    '''Loads the EV profiles of the simulation
    If load_from_replay_path is None, then the EV profiles are created randomly

    Returns:
        - ev_profiles: a list of ev_profile objects'''

    if env.load_from_replay_path is None:

        if env.scenario == 'GF':
            ev_profiles = EV_spawner_GF(env)
            while len(ev_profiles) == 0:
                ev_profiles = EV_spawner_GF(env)
            return ev_profiles

        ev_profiles = EV_spawner(env)
        while len(ev_profiles) == 0:
            ev_profiles = EV_spawner(env)

        return ev_profiles
    else:
        return env.replay.EVs


def load_electricity_prices(env) -> Tuple[np.ndarray, np.ndarray]:
    '''Loads the electricity prices of the simulation
    If load_from_replay_path is None, then the electricity prices are created randomly

    Returns:
        - charge_prices: a matrix of size (number of charging stations, simulation length) with the charge prices
        - discharge_prices: a matrix of size (number of charging stations, simulation length) with the discharge prices'''

    if env.load_from_replay_path is not None:
        return env.replay.charge_prices, env.replay.discharge_prices

    if env.price_data is None:
        # Allow overriding the default Netherlands dataset with a custom local CSV.
        price_data_file = env.config.get('price_data_file')
        if price_data_file in (None, 'None', ''):
            file_path = pkg_resources.resource_filename(
                'ev2gym', 'data/Netherlands_day-ahead-2015-2024.csv')
        else:
            file_path = price_data_file
            if not os.path.exists(file_path):
                try:
                    file_path = pkg_resources.resource_filename('ev2gym', price_data_file)
                except Exception as exc:
                    raise FileNotFoundError(
                        f'Price data file not found: {price_data_file}'
                    ) from exc

        env.price_data = pd.read_csv(file_path, sep=',', header=0)
        # import polars as pl
        # env.price_data = pl.read_csv(file_path).to_pandas()

        time_column = None
        for candidate in ['Datetime (UTC)', 'Timestamp']:
            if candidate in env.price_data.columns:
                time_column = candidate
                break
        if time_column is None:
            raise KeyError(
                f'No supported datetime column found in {file_path}. '
                'Expected one of: Datetime (UTC), Timestamp'
            )

        value_column = None
        for candidate in ['Price (EUR/MWhe)', 'Price_RMB_MWh']:
            if candidate in env.price_data.columns:
                value_column = candidate
                break
        if value_column is None:
            raise KeyError(
                f'No supported price column found in {file_path}. '
                'Expected one of: Price (EUR/MWhe), Price_RMB_MWh'
            )

        if value_column == 'Price_RMB_MWh':
            env.price_unit_label = 'RMB/kWh'
            env.profit_unit_label = 'RMB'
        else:
            env.price_unit_label = 'EUR/kWh'
            env.profit_unit_label = 'EUR'

        env.price_time_column = time_column
        env.price_value_column = value_column
        env.price_data[time_column] = pd.to_datetime(env.price_data[time_column])

        drop_columns = [col for col in ['Country', 'Datetime (Local)']
                        if col in env.price_data.columns]

        if drop_columns:
            env.price_data.drop(drop_columns, inplace=True, axis=1)
        env.price_data['year'] = pd.DatetimeIndex(
            env.price_data[time_column]).year
        env.price_data['month'] = pd.DatetimeIndex(
            env.price_data[time_column]).month
        env.price_data['day'] = pd.DatetimeIndex(
            env.price_data[time_column]).day
        env.price_data['hour'] = pd.DatetimeIndex(
            env.price_data[time_column]).hour
        env.price_data['minute'] = pd.DatetimeIndex(
            env.price_data[time_column]).minute

    # assume charge and discharge prices are the same
    # assume prices are the same for all charging stations

    data = env.price_data
    price_column = env.price_value_column
    use_minute_resolution = data['minute'].nunique() > 1
    charge_prices = np.zeros((env.cs, env.simulation_length))
    discharge_prices = np.zeros((env.cs, env.simulation_length))
    # for every simulation step, take the price of the corresponding hour
    sim_temp_date = env.sim_date
    for i in range(env.simulation_length):

        year = sim_temp_date.year
        month = sim_temp_date.month
        day = sim_temp_date.day
        hour = sim_temp_date.hour
        minute = sim_temp_date.minute
        mask = (
            (data['year'] == year) &
            (data['month'] == month) &
            (data['day'] == day) &
            (data['hour'] == hour)
        )
        if use_minute_resolution:
            mask = mask & (data['minute'] == minute)
        # find the corresponding price
        try:
            price = data.loc[mask, price_column].iloc[0] / 1000
            charge_prices[:, i] = -price
            discharge_prices[:, i] = price
        except:
            print(
                'Error: no price found for the given date and time. Using first available price instead.')

            price = data[price_column].iloc[0] / 1000
            charge_prices[:, i] = -price
            discharge_prices[:, i] = price

        # step to next
        sim_temp_date = sim_temp_date + \
            datetime.timedelta(minutes=env.timescale)

    discharge_prices = discharge_prices * env.config['discharge_price_factor']
    return charge_prices, discharge_prices


def load_grid(env):
    '''Loads the grid of the simulation'''

    if env.load_from_replay_path is not None:
        env.cs_transformers = env.replay.cs_transformers
        return env.replay.grid

    # Simulate grid
    if env.simulate_grid:
        if env.load_from_replay_path is None:
            pv_profile = load_pv_profiles(env)
            
        grid = PowerGrid(env.config,
                         env=env,
                         pv_profile=pv_profile,
                         )

        env.number_of_transformers = grid.node_num-1
        env.cs_transformers = [
            *np.arange(env.number_of_transformers)] * (env.cs // env.number_of_transformers)
        env.cs_transformers += np.arange(
            env.cs % env.number_of_transformers).tolist()
        # print(f'cs_transformers: {env.cs_transformers}')

        assert env.charging_network_topology is None, "Charging network topology is not supported with grid simulation."

        

        return grid

    if env.charging_network_topology is None:
        env.cs_transformers = [
            *np.arange(env.number_of_transformers)] * (env.cs // env.number_of_transformers)
        env.cs_transformers += np.arange(
            env.cs % env.number_of_transformers).tolist()

    return None


def load_pv_profiles(env) -> np.ndarray:

    # Load the data
    data_path = pkg_resources.resource_filename(
        'ev2gym', 'data/pv_netherlands.csv')
    data = pd.read_csv(data_path, sep=',', header=0)
    data.drop(['time', 'local_time'], inplace=True, axis=1)

    desired_timescale = env.timescale

    dataset_timescale = 60
    dataset_starting_date = '2019-01-01 00:00:00'

    if desired_timescale > dataset_timescale:
        data = data.groupby(
            data.index // (desired_timescale/dataset_timescale)).max()
    elif desired_timescale < dataset_timescale:
        data = data.loc[data.index.repeat(
            dataset_timescale/desired_timescale)].reset_index(drop=True)

    # smooth data by taking the mean of every 5 rows
    data['electricity'] = data['electricity'].rolling(
        window=60//desired_timescale, min_periods=1).mean()
    # use other type of smoothing
    data['electricity'] = data['electricity'].ewm(
        span=60//desired_timescale, adjust=True).mean()

    # duplicate the data to have two years of data
    data = pd.concat([data, data], ignore_index=True)

    # add a date column to the dataframe
    data['date'] = pd.date_range(
        start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')

    return data
