"""Rutas de archivos del proyecto. Unica capa que conoce el sistema de archivos."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODELS_DIR = OUTPUT_DIR / "models"

MATCHES_CSV = DATA_DIR / "matches.csv"
TEAMS_CSV = DATA_DIR / "teams.csv"
MATCH_TEAM_STATS_CSV = DATA_DIR / "match_team_stats.csv"
RESULTS_CSV = DATA_DIR / "results.csv"
SHOOTOUTS_CSV = DATA_DIR / "shootouts.csv"
SQUADS_CSV = DATA_DIR / "squads_and_players.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
