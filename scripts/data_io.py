#!/usr/bin/env python3
"""
Utilities for loading and joining air quality and weather CSV files.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


AIR_GLOB = "AirBishkek*.csv"
WEATHER_GLOB = "WeatherBishkek*.csv"
AIR_DATETIME_FORMAT = "%Y-%m-%d %I:%M %p"
WEATHER_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "csvs"


def extract_year(path: Path) -> int | None:
    match = re.search(r"(19|20)\d{2}", path.stem)
    if not match:
        return None
    return int(match.group(0))


def list_csvs(data_dir: Path, pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(Path(data_dir) / pattern)))


def parse_air_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format=AIR_DATETIME_FORMAT, errors="coerce")


def parse_weather_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format=WEATHER_DATETIME_FORMAT, errors="coerce")


def format_weather_datetime(values: pd.Series) -> pd.Series:
    return values.dt.strftime(WEATHER_DATETIME_FORMAT)


def load_air_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Parameter"].str.contains("PM2.5", na=False)].copy()
    df["datetime"] = parse_air_datetime(df["Date (LT)"])
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
    df["datetime"] = parse_weather_datetime(df["datetime"])
    categorical_cols = {"name", "datetime", "preciptype", "conditions", "icon", "stations"}
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.drop_duplicates(subset=["datetime"])
    return df


def pair_years(
    air_paths: list[Path],
    weather_paths: list[Path],
    *,
    strict: bool = False,
) -> dict[int, tuple[Path, Path]]:
    air_by_year = {year: path for path in air_paths if (year := extract_year(path)) is not None}
    weather_by_year = {
        year: path for path in weather_paths if (year := extract_year(path)) is not None
    }

    years = sorted(set(air_by_year) & set(weather_by_year))
    if not years:
        raise FileNotFoundError("No matching air/weather year pairs found.")

    if strict:
        missing_air = sorted(set(weather_by_year) - set(air_by_year))
        missing_weather = sorted(set(air_by_year) - set(weather_by_year))
        if missing_air or missing_weather:
            raise FileNotFoundError(
                f"Missing pairs. No air for years: {missing_air}. No weather for years: {missing_weather}."
            )

    return {year: (air_by_year[year], weather_by_year[year]) for year in years}


def merge_yearly_air_weather(air_path: Path, weather_path: Path) -> pd.DataFrame:
    air = load_air_csv(air_path)
    weather = load_weather_csv(weather_path)
    merged = air.merge(weather, on="datetime", how="inner")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["datetime"] = format_weather_datetime(merged["datetime"])
    return merged


def build_merged_table_by_year(
    data_dir: Path,
    air_pattern: str = AIR_GLOB,
    weather_pattern: str = WEATHER_GLOB,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    air_paths = list_csvs(data_dir, air_pattern)
    weather_paths = list_csvs(data_dir, weather_pattern)

    if not air_paths:
        raise FileNotFoundError(f"No air CSV files found in {data_dir}")
    if not weather_paths:
        raise FileNotFoundError(f"No weather CSV files found in {data_dir}")

    frames: list[pd.DataFrame] = []
    for year, (air_path, weather_path) in pair_years(air_paths, weather_paths, strict=strict).items():
        merged = merge_yearly_air_weather(air_path, weather_path)
        merged["data_year"] = year
        frames.append(merged)

    merged_all = pd.concat(frames, ignore_index=True)
    merged_all = merged_all.sort_values("datetime").reset_index(drop=True)
    return merged_all


def build_merged_table(
    data_dir: Path,
    air_pattern: str = AIR_GLOB,
    weather_pattern: str = WEATHER_GLOB,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    return build_merged_table_by_year(
        data_dir,
        air_pattern=air_pattern,
        weather_pattern=weather_pattern,
        strict=strict,
    )
