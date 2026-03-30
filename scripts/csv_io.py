#!/usr/bin/env python3
"""
Utilities for loading and joining air quality and weather CSV files.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "csvs"
DATA_YEARS = [2019, 2020, 2021, 2022, 2023]
AIR_PATTERN = "AirBishkek*.csv"
WEATHER_PATTERN = "WeatherBishkek*.csv"
AIR_DATETIME_FORMAT = "%Y-%m-%d %I:%M %p"
WEATHER_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

def load_air_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Parameter"].str.contains("PM2.5", na=False)].copy()
    df["datetime"] = pd.to_datetime(df["Date (LT)"], format=AIR_DATETIME_FORMAT, errors="coerce")
    df["pm25"] = pd.to_numeric(df["Raw Conc."], errors="coerce")
    df["qc_name"] = df["QC Name"].astype(str)
    df.loc[df["pm25"] <= -900, "pm25"] = np.nan
    df = df[df["qc_name"].str.lower() == "valid"]
    df = df[["datetime", "pm25"]]
    df = df.dropna(subset=["datetime", "pm25"])
    df = df.drop_duplicates(subset=["datetime"])
    return df


def load_weather_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], format=WEATHER_DATETIME_FORMAT, errors="coerce")
    categorical_cols = {"name", "datetime", "preciptype", "conditions", "icon", "stations"}
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.drop_duplicates(subset=["datetime"])
    return df


def merge_csvs(paired_paths: tuple[Path, Path]) -> pd.DataFrame:
    if not paired_paths[0].exists() or not paired_paths[1].exists():
        raise FileNotFoundError(
            f"Either of {paired_paths[0]} or {paired_paths[1]} was not found!"
        )

    air = load_air_csv(paired_paths[0])
    weather = load_weather_csv(paired_paths[1])

    merged = air.merge(weather, on="datetime", how="inner")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["datetime"] = merged["datetime"].dt.strftime(WEATHER_DATETIME_FORMAT)
    
    return merged


def build_ultimate_table() -> pd.DataFrame:
    paired_paths: list[(Path, Path)] = []
    for year in DATA_YEARS:
        air_path = DATA_DIR / AIR_PATTERN.replace("*", str(year))
        weather_path = DATA_DIR / WEATHER_PATTERN.replace("*", str(year))
        paired_paths.append((air_path, weather_path))
    
    merged_tables:list[pd.DataFrame] = []
    for pp in paired_paths:
        merged_tables.append(merge_csvs(pp))
        
    merged_all: pd.DataFrame = pd.concat(merged_tables, ignore_index=True)
    merged_all = merged_all.sort_values("datetime").reset_index(drop=True)
    merged_all.to_csv("out.csv", index=False)

    return merged_all


if __name__ == "__main__":
    build_ultimate_table()
