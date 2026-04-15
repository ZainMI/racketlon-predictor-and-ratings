from pathlib import Path
from typing import Any, Dict, Optional

# raw data pipeline
from match_scraper import main as scrape_main
from data_clean import clean_matches
from features import build_training_data

# model service
from models.catboost.catboost_fin import (
    train_full_and_package,
    load_predictor,
    predict_match,
    get_player_ratings,
)

DEFAULT_MATCHES_CSV = "data/matches.csv"
DEFAULT_MATCHES_CLEAN_CSV = "data/matches_cleaned.csv"
DEFAULT_DATA_CSV = "data/data.csv"
DEFAULT_INFERENCE_STATE = "data/inference_state.pkl"
DEFAULT_MODEL_DIR = "models/catboost/artifacts/predictor_package"


_predictor_cache = None
_predictor_cache_dir = None


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _normalize_player_name(name: str) -> str:
    return str(name).strip().lower()


def clear_model_cache() -> None:
    global _predictor_cache, _predictor_cache_dir
    _predictor_cache = None
    _predictor_cache_dir = None


def ensure_data_dir() -> None:
    Path("data").mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Step 1: scrape
# -------------------------------------------------
def scrape_matches() -> Dict[str, Any]:
    """
    Runs the scraper exactly as if you called match_scraper.py directly.
    Assumes match_scraper.py writes to data/matches.csv or its configured output.
    """
    scrape_main()
    return {
        "status": "ok",
        "message": "Scraping complete.",
    }


# -------------------------------------------------
# Step 2: clean
# -------------------------------------------------
def clean_match_data(
    input_csv: str = DEFAULT_MATCHES_CSV,
    output_csv: str = DEFAULT_MATCHES_CLEAN_CSV,
) -> Dict[str, Any]:
    ensure_data_dir()
    clean_matches(input_csv=input_csv, output_csv=output_csv)
    return {
        "status": "ok",
        "input_csv": input_csv,
        "output_csv": output_csv,
    }


# -------------------------------------------------
# Step 3: build training features
# -------------------------------------------------
def build_feature_data(
    in_csv: str = DEFAULT_MATCHES_CLEAN_CSV,
    out_csv: str = DEFAULT_DATA_CSV,
    out_inference_state: str = DEFAULT_INFERENCE_STATE,
) -> Dict[str, Any]:
    ensure_data_dir()
    build_training_data(
        in_csv=in_csv,
        out_csv=out_csv,
        out_inference_state=out_inference_state,
    )
    return {
        "status": "ok",
        "input_csv": in_csv,
        "data_csv": out_csv,
        "inference_state": out_inference_state,
    }


# -------------------------------------------------
# Step 4: train full model
# -------------------------------------------------
def train_model(
    data_path: str = DEFAULT_DATA_CSV,
    inference_state_path: str = DEFAULT_INFERENCE_STATE,
    output_dir: str = DEFAULT_MODEL_DIR,
    reload_after_train: bool = True,
) -> Dict[str, Any]:
    predictor = train_full_and_package(
        data_path=data_path,
        inference_state_path=inference_state_path,
        output_dir=output_dir,
    )

    if reload_after_train:
        clear_model_cache()
        load_model(output_dir=output_dir, force_reload=True)

    return {
        "status": "ok",
        "message": "Model trained on full data and saved.",
        "output_dir": output_dir,
        "metadata": predictor.metadata,
    }


# -------------------------------------------------
# Load / cache model
# -------------------------------------------------
def load_model(
    output_dir: str = DEFAULT_MODEL_DIR,
    force_reload: bool = False,
):
    global _predictor_cache, _predictor_cache_dir

    if (
        not force_reload
        and _predictor_cache is not None
        and _predictor_cache_dir == output_dir
    ):
        return _predictor_cache

    predictor = load_predictor(output_dir)
    _predictor_cache = predictor
    _predictor_cache_dir = output_dir
    return predictor


# -------------------------------------------------
# Predict
# -------------------------------------------------
def predict(
    player1: str,
    player2: str,
    output_dir: str = DEFAULT_MODEL_DIR,
    force_reload: bool = False,
) -> Dict[str, Any]:
    predictor = load_model(output_dir=output_dir, force_reload=force_reload)
    return predict_match(
        predictor,
        _normalize_player_name(player1),
        _normalize_player_name(player2),
    )


def player_state(
    player: str,
    output_dir: str = DEFAULT_MODEL_DIR,
    force_reload: bool = False,
) -> Dict[str, Any]:
    predictor = load_model(output_dir=output_dir, force_reload=force_reload)
    return get_player_ratings(
        predictor,
        _normalize_player_name(player),
    )


def matchup_bundle(
    player1: str,
    player2: str,
    output_dir: str = DEFAULT_MODEL_DIR,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """
    Convenient API response:
    - match prediction
    - player 1 state
    - player 2 state
    """
    predictor = load_model(output_dir=output_dir, force_reload=force_reload)

    p1 = _normalize_player_name(player1)
    p2 = _normalize_player_name(player2)

    return {
        "prediction": predict_match(predictor, p1, p2),
        "player1_state": get_player_ratings(predictor, p1),
        "player2_state": get_player_ratings(predictor, p2),
    }


# -------------------------------------------------
# Full rebuild pipeline
# -------------------------------------------------
def rebuild_all(
    scrape: bool = False,
    matches_csv: str = DEFAULT_MATCHES_CSV,
    matches_clean_csv: str = DEFAULT_MATCHES_CLEAN_CSV,
    data_csv: str = DEFAULT_DATA_CSV,
    inference_state_path: str = DEFAULT_INFERENCE_STATE,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
      optionally scrape -> clean -> feature build -> full train
    """
    ensure_data_dir()

    if scrape:
        scrape_main()

    clean_matches(
        input_csv=matches_csv,
        output_csv=matches_clean_csv,
    )

    build_training_data(
        in_csv=matches_clean_csv,
        out_csv=data_csv,
        out_inference_state=inference_state_path,
    )

    predictor = train_full_and_package(
        data_path=data_csv,
        inference_state_path=inference_state_path,
        output_dir=model_dir,
    )

    clear_model_cache()
    load_model(output_dir=model_dir, force_reload=True)

    return {
        "status": "ok",
        "message": "Full rebuild complete.",
        "matches_clean_csv": matches_clean_csv,
        "data_csv": data_csv,
        "inference_state": inference_state_path,
        "model_dir": model_dir,
        "metadata": predictor.metadata,
    }
