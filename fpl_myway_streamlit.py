"""
Streamlit App for FPLMyWay
================================

This module implements a basic user interface for the Fantasy Premier League
assistant described by the user.  The application is built around the
``streamlit`` framework and demonstrates how to load data from the official
FPL API as well as an open understat data set for expected goals and
assists.  Users can interact with the app using free‑form natural
language commands or via the provided controls.  The goal of this
application is not to produce the optimal team for every situation but
rather to show how such a tool could be structured using free data
sources and off‑the‑shelf Python libraries.

Key features implemented in this example include:

* Loading and caching FPL data downloaded via the browser (``download.json``
  and ``download (1).json`` in the working directory).  The app expects
  that these files have already been downloaded using the browser as
  described in the assistant’s workflow.  The bootstrap file contains
  information about all FPL players and teams, while the fixtures file
  contains the upcoming match schedule.  To keep the app free from
  external dependencies, network access from Python is avoided.
* Loading an example Understat CSV file (``players_epl_21-22.csv``) that
  provides shot and expected goal information for players in the English
  Premier League.  Where available this information is merged onto the
  FPL player table to augment the decision making.  Because naming
  conventions between the two data sets differ slightly, only approximate
  matches are performed using case‑insensitive player names.
* A simple points projection model based on ``points_per_game`` from the
  FPL data and a user specified horizon (number of gameweeks).  Users
  can override the horizon via natural language (e.g. “for the next 5
  GWs”).  Predicted points are multiplied by the horizon and converted
  into a “value” score based on the player’s price.
* A greedy team builder that constructs a 15‑man squad given budget
  constraints, positional requirements (2 goalkeepers, 5 defenders,
  5 midfielders, 3 forwards) and a maximum of three players per club.
  While not a full linear programming optimiser, this approach
  demonstrates the mechanics of balancing positions, budget and team
  limits.
* A rudimentary command parser that recognises a handful of phrases
  including “suggest best transfers”, “rate my team”, “wildcard draft” and
  “differential”.  If no recognised command is detected the app
  displays help text summarising the available functionality.

To run this app locally you will need to have ``streamlit`` installed
(``pip install streamlit``) and then execute ``streamlit run
fpl_myway_streamlit.py`` from the command line.  On systems without
network access to pip, ``streamlit`` can be installed on a
cloud‑hosted service such as Streamlit Community Cloud or Replit, both
of which offer free tiers.  The user’s downloaded data files should be
placed in the same directory as this script or the paths adjusted
accordingly.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st  # type: ignore
except ImportError:
    # When streamlit is not installed locally the module will still import
    # successfully for type checkers.  The actual app must be run in an
    # environment where streamlit is available (e.g. Streamlit Cloud).
    st = None  # type: ignore


###############################################################################
# Data loading functions
###############################################################################

@st.cache_data if st else (lambda func: func)
def load_fpl_data(bootstrap_path: Path, fixtures_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """Load the FPL bootstrap and fixtures JSON files.

    Parameters
    ----------
    bootstrap_path : Path
        Path to the ``download.json`` file containing the bootstrap data.
    fixtures_path : Path
        Path to the ``download (1).json`` file containing fixture data.

    Returns
    -------
    players_df : pd.DataFrame
        Data frame of FPL players with relevant fields.
    teams_df : pd.DataFrame
        Data frame of FPL teams with relevant fields.
    fixtures_df : pd.DataFrame
        Data frame of FPL fixtures with relevant fields.
    position_map : dict
        Mapping from element_type id to human readable position names.
    """
    with open(bootstrap_path) as f:
        bootstrap = json.load(f)

    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    element_types = pd.DataFrame(bootstrap["element_types"])

    # Select only columns that we care about for this prototype
    players_df = players[
        [
            "id",
            "first_name",
            "second_name",
            "web_name",
            "team",
            "element_type",
            "now_cost",
            "points_per_game",
            "selected_by_percent",
            "chance_of_playing_next_round",
            "expected_goals_per_90",
            "expected_assists_per_90",
            "minutes",
            "goals_scored",
            "assists",
            "total_points",
        ]
    ].copy()

    # Convert data types
    players_df["points_per_game"] = pd.to_numeric(players_df["points_per_game"], errors="coerce").fillna(0.0)
    players_df["selected_by_percent"] = pd.to_numeric(players_df["selected_by_percent"], errors="coerce").fillna(0.0)
    players_df["now_cost"] = players_df["now_cost"].astype(float)
    players_df["expected_goals_per_90"] = pd.to_numeric(players_df["expected_goals_per_90"], errors="coerce").fillna(0.0)
    players_df["expected_assists_per_90"] = pd.to_numeric(players_df["expected_assists_per_90"], errors="coerce").fillna(0.0)

    # Build position mapping
    position_map = {
        row["id"]: row["singular_name"] for _, row in element_types[["id", "singular_name"]].iterrows()
    }
    players_df["position"] = players_df["element_type"].map(position_map)

    # Load fixtures
    with open(fixtures_path) as f:
        fixtures_raw = json.load(f)
    fixtures_df = pd.DataFrame(fixtures_raw)

    return players_df, teams, fixtures_df, position_map


@st.cache_data if st else (lambda func: func)
def load_understat_data(understat_path: Path) -> pd.DataFrame:
    """Load Understat player level data from a CSV file.

    The Understat data used here is a static snapshot of the 2021–22
    Premier League season.  Columns include expected goals (xG), assists
    (xA), shots, key passes and other summary statistics.  When
    combining with FPL data the app performs a simple case‑insensitive
    match on the player name.
    """
    df = pd.read_csv(understat_path)
    # Normalise names by stripping and lowercasing
    df["player_name_norm"] = df["player_name"].str.replace(" ", "").str.lower()
    return df


###############################################################################
# Utility functions
###############################################################################

def merge_understat(players_df: pd.DataFrame, understat_df: pd.DataFrame) -> pd.DataFrame:
    """Merge FPL players with Understat expected goal data.

    Understat and FPL players are matched using a simplified normalised
    version of the name (concatenation of first and last names in
    lowercase).  Players without Understat data remain unchanged.

    Parameters
    ----------
    players_df : pd.DataFrame
        Data frame containing FPL players.
    understat_df : pd.DataFrame
        Understat player data.

    Returns
    -------
    players_df : pd.DataFrame
        Input data frame augmented with columns from Understat (prefixed
        with ``u_``) where available.
    """
    players_df = players_df.copy()
    players_df["name_norm"] = (
        players_df["first_name"].str.cat(players_df["second_name"], sep="")
    ).str.replace(" ", "").str.lower()
    merge_df = pd.merge(
        players_df,
        understat_df,
        how="left",
        left_on="name_norm",
        right_on="player_name_norm",
        suffixes=("", "_u"),
    )
    # Prefix Understat columns to avoid clashes
    understat_cols = [
        col for col in understat_df.columns if col not in {"player_name", "player_name_norm"}
    ]
    for col in understat_cols:
        new_col = f"u_{col}"
        merge_df[new_col] = merge_df[col]
    return merge_df


def compute_predicted_points(
    df: pd.DataFrame, horizon: int, use_expected: bool = False
) -> pd.DataFrame:
    """Compute projected points over a number of gameweeks.

    Two simple scoring models are provided.  The default model uses
    ``points_per_game`` from the FPL data and multiplies it by the
    horizon.  The optional ``use_expected`` model computes points from
    expected goals and assists per 90 minutes assuming 4 points per goal
    and 3 points per assist for all players irrespective of position.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with FPL player information.
    horizon : int
        Number of upcoming gameweeks to project.
    use_expected : bool, optional
        If True, use expected goals and assists per 90 minutes to
        compute projected points.  Otherwise use historical points per
        game.

    Returns
    -------
    pd.DataFrame
        A copy of the input data with an additional column
        ``predicted_points``.
    """
    df = df.copy()
    if use_expected:
        # Points formula: 4 points per goal + 3 points per assist
        # Use expected values per 90 minutes and the player's average minutes
        df["predicted_points"] = (
            (df["expected_goals_per_90"] * 4 + df["expected_assists_per_90"] * 3)
            * (df["minutes"] / 90.0)
        )
        df["predicted_points"] = df["predicted_points"].fillna(0) * horizon
    else:
        df["predicted_points"] = df["points_per_game"] * horizon
    return df


def build_squad(
    player_pool: pd.DataFrame,
    budget: float = 100.0,
    horizon: int = 3,
    max_players_per_team: int = 3,
    differential_weight: float = 0.0,
    preseason_weight: float = 0.0,
    position_requirements: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Construct a 15‑player squad using a greedy selection algorithm.

    Players are first scored using projected points divided by price.
    Additional weighting can be applied to low ownership players by
    increasing ``differential_weight`` and to Understat pre‑season xG
    (if available) by increasing ``preseason_weight``.

    Parameters
    ----------
    player_pool : pd.DataFrame
        Data frame of candidate players with predicted points and price.
    budget : float, optional
        Total budget available in millions (default 100.0).
    horizon : int, optional
        Number of gameweeks used when computing predicted points.
    max_players_per_team : int, optional
        Maximum number of players from any single club.
    differential_weight : float, optional
        Weighting factor to boost the score of players with low
        ownership.  The weight is multiplied by ``(100 - selected_by_percent)``.
    preseason_weight : float, optional
        Weighting factor to boost the score of players with high
        Understat xG values (using the ``u_xG`` column if present).
    position_requirements : dict, optional
        Number of required players by position.  Defaults to the
        standard FPL requirements.

    Returns
    -------
    pd.DataFrame
        Data frame containing the selected squad.
    """
    if position_requirements is None:
        position_requirements = {
            "Goalkeeper": 2,
            "Defender": 5,
            "Midfielder": 5,
            "Forward": 3,
        }

    # Copy to avoid modifying original
    pool = player_pool.copy()
    # Compute base score
    pool["value"] = pool["now_cost"] / 10.0
    pool["score"] = pool["predicted_points"] / pool["value"]
    # Apply differential weighting (players with low ownership receive a boost)
    if differential_weight > 0:
        pool["score"] += differential_weight * (100 - pool["selected_by_percent"])
    # Apply preseason weighting if Understat data available
    if preseason_weight > 0 and "u_xG" in pool.columns:
        pool["score"] += preseason_weight * pool["u_xG"]

    # Sort by score descending
    pool = pool.sort_values(by="score", ascending=False)

    selected_indices: List[int] = []
    team_counts: Dict[int, int] = {team_id: 0 for team_id in pool["team"].unique()}
    position_counts: Dict[str, int] = {pos: 0 for pos in position_requirements.keys()}
    total_cost = 0.0

    for idx, row in pool.iterrows():
        pos = row["position"]
        team_id = row["team"]
        cost = row["value"]
        # Check position quota
        if position_counts[pos] >= position_requirements[pos]:
            continue
        # Check team quota
        if team_counts[team_id] >= max_players_per_team:
            continue
        # Check budget
        if total_cost + cost > budget:
            continue
        # Add player
        selected_indices.append(idx)
        position_counts[pos] += 1
        team_counts[team_id] += 1
        total_cost += cost
        # Stop if we have 15 players
        if len(selected_indices) == sum(position_requirements.values()):
            break
    return pool.loc[selected_indices].reset_index(drop=True)


def parse_command(command: str) -> Dict[str, any]:
    """Parse a natural language command into structured parameters.

    This function recognises a handful of phrases and extracts options
    such as the horizon (number of future gameweeks), positional
    constraints and differential preferences.  The returned dictionary
    contains the keys needed by the downstream logic.  If a command
    cannot be parsed the function returns an empty dictionary.

    Parameters
    ----------
    command : str
        Raw command entered by the user.

    Returns
    -------
    dict
        Parsed parameters.  Keys may include ``horizon``,
        ``type`` ("transfers", "wildcard", "rate", etc.),
        ``max_from_team`` and ``differential``.
    """
    cmd = command.lower()
    params: Dict[str, any] = {}
    # Horizon detection e.g. "next 3 gws" or "next three gameweeks"
    horizon_match = re.search(r"next\s+(\d+)\s*g[aw]", cmd)
    if horizon_match:
        try:
            params["horizon"] = int(horizon_match.group(1))
        except ValueError:
            pass
    # Command type detection
    if "transfer" in cmd:
        params["type"] = "transfers"
    elif "wildcard" in cmd:
        params["type"] = "wildcard"
    elif "rate" in cmd:
        params["type"] = "rate"
    # Differential detection
    if "differential" in cmd:
        params["differential"] = True
    # Team constraint e.g. "only 2 players from liverpool"
    team_match = re.search(r"only\s+(\d+)\s+players?\s+from\s+([a-zA-Z ]+)", cmd)
    if team_match:
        try:
            params["team_limit"] = int(team_match.group(1))
            params["team_name"] = team_match.group(2).strip().title()
        except ValueError:
            pass
    return params


def respond_to_command(
    params: Dict[str, any],
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    position_map: Dict[int, str],
    horizon_default: int = 3,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """Generate a human readable response to a parsed command.

    Based on the parsed parameters, this function delegates to the
    appropriate helper functions to build a squad, suggest transfers or
    rate an existing team.  It returns a message string along with an
    optional data frame for display.
    """
    # Determine horizon
    horizon = params.get("horizon", horizon_default)
    cmd_type = params.get("type", "help")
    if cmd_type == "transfers":
        # Suggest top players to transfer in
        df = compute_predicted_points(players_df, horizon)
        # Exclude players flagged as injured or unavailable
        df = df[df["chance_of_playing_next_round"].fillna(100) > 25]
        # Sort by predicted points
        df = df.sort_values(by="predicted_points", ascending=False)
        suggestions = df.head(10)[[
            "web_name",
            "position",
            "team",
            "now_cost",
            "predicted_points",
        ]]
        # Map team ids to team names
        suggestions["team"] = suggestions["team"].apply(
            lambda tid: teams_df.loc[teams_df["id"] == tid, "name"].values[0]
        )
        msg = f"Top transfer targets for the next {horizon} gameweeks:"
        return msg, suggestions
    elif cmd_type == "wildcard":
        # Build a wildcard draft
        team_limit = params.get("team_limit", 3)
        specific_team_name = params.get("team_name")
        df = compute_predicted_points(players_df, horizon)
        # If the user specified a particular team limit for one club, override
        if specific_team_name:
            # Map team name to id
            matching = teams_df[teams_df["name"].str.contains(specific_team_name, case=False)]
            if not matching.empty:
                team_id = matching.iloc[0]["id"]
                # Create a copy of player pool where team limit is applied only
                # to the specified team; others remain at default 3.
                squad = build_squad(
                    df,
                    budget=100.0,
                    horizon=horizon,
                    max_players_per_team=3,
                )
                # Filter squad to ensure constraint on specific team
                counts = squad["team"].value_counts().to_dict()
                while counts.get(team_id, 0) > team_limit:
                    # Remove the lowest scoring player from that team
                    idx_to_drop = (
                        squad[squad["team"] == team_id]["score"].idxmin()
                    )
                    squad = squad.drop(idx_to_drop)
                    counts = squad["team"].value_counts().to_dict()
                # Refill squad with best available players until 15 players
                needed = 15 - len(squad)
                if needed > 0:
                    excluded_ids = set(squad["id"])
                    candidate_pool = df[~df["id"].isin(excluded_ids)]
                    top_candidates = candidate_pool.sort_values(by="score", ascending=False)
                    for _, row in top_candidates.iterrows():
                        if len(squad) >= 15:
                            break
                        if counts.get(row["team"], 0) >= 3:
                            continue
                        # Add row
                        squad = pd.concat([squad, row.to_frame().T], ignore_index=True)
                        counts[row["team"]] = counts.get(row["team"], 0) + 1
                # Map team ids to names
                squad_display = squad[["web_name", "position", "team", "value", "predicted_points"]].copy()
                squad_display["team"] = squad_display["team"].apply(
                    lambda tid: teams_df.loc[teams_df["id"] == tid, "name"].values[0]
                )
                msg = f"Wildcard draft (max {team_limit} from {specific_team_name})"
                return msg, squad_display
        # Default wildcard without specific team limit
        squad = build_squad(df, budget=100.0, horizon=horizon)
        squad_display = squad[["web_name", "position", "team", "value", "predicted_points"]].copy()
        squad_display["team"] = squad_display["team"].apply(
            lambda tid: teams_df.loc[teams_df["id"] == tid, "name"].values[0]
        )
        msg = f"Wildcard draft for the next {horizon} gameweeks:"
        return msg, squad_display
    elif cmd_type == "rate":
        # Rate a team provided by the user.  Expect comma separated names after the command.
        # Extract names after colon or the word "team"
        m = re.search(r":\s*(.*)", command)
        names_str = m.group(1) if m else ""
        names = [n.strip().lower() for n in names_str.split(",") if n.strip()]
        if not names:
            return (
                "To rate your team, list the players after a colon, e.g. 'rate my team: Salah, Saka, Haaland'.",
                None,
            )
        df = compute_predicted_points(players_df, horizon)
        # Match players by web_name ignoring case
        rating_df = df[df["web_name"].str.lower().isin(names)].copy()
        missing = [n for n in names if n not in rating_df["web_name"].str.lower().tolist()]
        if missing:
            msg = f"The following players were not recognised: {', '.join(missing)}"
        else:
            msg = f"Projected points for your team over the next {horizon} gameweeks."
        rating_display = rating_df[["web_name", "position", "team", "now_cost", "predicted_points"]].copy()
        rating_display["team"] = rating_display["team"].apply(
            lambda tid: teams_df.loc[teams_df["id"] == tid, "name"].values[0]
        )
        return msg, rating_display
    else:
        # Help / default message
        help_text = (
            "Enter a command such as:\n\n"
            "- 'Suggest best transfers for next 3 GWs' to see the top performers.\n"
            "- 'Wildcard draft with only 2 players from Liverpool' to build a draft squad.\n"
            "- 'Rate my team: Salah, Saka, Haaland' to get a projection for your team.\n\n"
            "You can also mention differentials in your command to prioritise low ownership players."
        )
        return help_text, None


###############################################################################
# Main Streamlit application
###############################################################################

def main() -> None:
    """Run the Streamlit application."""
    if st is None:
        raise ImportError(
            "This application requires streamlit. Please install it with 'pip install streamlit' and run via 'streamlit run fpl_myway_streamlit.py'."
        )

    st.set_page_config(page_title="FPLMyWay", layout="wide")
    st.title("FPLMyWay – Speak to the Model, Shape Your Squad")

    st.write(
        "This tool uses open Fantasy Premier League data and Understat xG statistics to "
        "provide high level guidance for squad selection and transfer planning.  "
        "It does **not** guarantee optimal decisions but can help you explore options."
    )

    # File paths relative to the script
    bootstrap_path = Path(__file__).parent / "download.json"
    fixtures_path = Path(__file__).parent / "download (1).json"
    understat_path = Path(__file__).parent / "players_epl_21-22.csv"

    # Load data
    players_df, teams_df, fixtures_df, position_map = load_fpl_data(
        bootstrap_path, fixtures_path
    )
    understat_df = load_understat_data(understat_path)
    players_df = merge_understat(players_df, understat_df)

    # Sidebar controls
    st.sidebar.header("Configuration")
    horizon = st.sidebar.slider("Projection horizon (gameweeks)", min_value=1, max_value=8, value=3, step=1)
    use_expected = st.sidebar.checkbox(
        "Use expected goals/assists instead of points per game", value=False
    )
    differential_weight = st.sidebar.slider(
        "Differential weighting", min_value=0.0, max_value=1.0, value=0.0, step=0.1
    )
    preseason_weight = st.sidebar.slider(
        "Pre‑season (Understat xG) weighting", min_value=0.0, max_value=1.0, value=0.0, step=0.1
    )

    command = st.text_input(
        "Enter a command (e.g. 'Suggest best transfers for next 3 GWs')", value=""
    )
    # Process command when the user presses enter or clicks the button
    if st.button("Run command"):
        params = parse_command(command)
        # Apply sidebar overrides
        params.setdefault("horizon", horizon)
        params.setdefault("differential_weight", differential_weight)
        params.setdefault("preseason_weight", preseason_weight)
        # Update predicted points based on selected model
        player_pool = compute_predicted_points(players_df, params["horizon"], use_expected)
        # Save score to use in wildcard building
        player_pool = player_pool.copy()
        player_pool["value"] = player_pool["now_cost"] / 10.0
        player_pool["score"] = player_pool["predicted_points"] / player_pool["value"]
        # Respond
        msg, data = respond_to_command(
            params,
            player_pool,
            teams_df,
            fixtures_df,
            position_map,
            horizon_default=horizon,
        )
        st.subheader("Response")
        st.write(msg)
        if data is not None:
            st.dataframe(data)

    # Display data summary when no command is run
    if command.strip() == "":
        st.subheader("Player Pool Summary")
        st.write(
            "The table below shows a few sample players from the FPL data. "
            "Use the command box above to start exploring transfers, drafts and ratings."
        )
        sample_df = compute_predicted_points(players_df, horizon, use_expected).head(20)[
            [
                "web_name",
                "position",
                "team",
                "now_cost",
                "points_per_game",
                "selected_by_percent",
                "predicted_points",
            ]
        ].copy()
        sample_df["team"] = sample_df["team"].apply(
            lambda tid: teams_df.loc[teams_df["id"] == tid, "name"].values[0]
        )
        st.dataframe(sample_df)


if __name__ == "__main__":
    main()