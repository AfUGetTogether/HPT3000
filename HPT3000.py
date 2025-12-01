import io
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import streamlit as st
import geopandas as gpd
import pandas as pd
import json

# Numerik/Geometrie & Akustik
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.strtree import STRtree
from shapely import prepared
from shapely import union_all
from shapely import ops

import shapely  # noqa: F401  (wird implizit gebraucht)
import sys, inspect  # noqa: F401

import pydeck as pdk

# ---------- Page config ----------
st.set_page_config(page_title="Heatpump-Positioning-Tool 3000", layout="wide")
st.title("Heatpump-Positioning-Tool 3000")

# ==== Intro ============================================================
with st.expander("Was macht dieses Tool? – aufklappen", expanded=False):
    st.markdown("""
**Das HPT3000** hat jetzt zwei Betriebsmodi:

1. **Optimaler Standort je Gebäude**  
   Für jedes Gebäude wird ein Wärmepumpen-Standort gesucht (zuerst auf dem Boden, optional auf Dachflächen), so dass:
   - alle **Nachbargebäude** ihre zulässigen Fassadenpegel einhalten (inkl. Sicherheitsaufschlag),
   - an der **eigenen Fassade** (Bodenaufstellung) optional ein **Eigen-Immissionszuschlag** eingehalten wird,
   - nahe Wände ≤ 3 m mit Sichtlinie automatisch **+3 / +6 dB** berücksichtigen,
   - Dachaufstellung (z. B. Flachdach) als Fallback genutzt werden kann (ohne Wandaufschlag, ohne Eigen-Grenze).

2. **Rasterbewertung aller Zellen**  
   Statt eines einzelnen optimalen Punktes werden alle Rasterpunkte auf:
   - der **Bodenfläche** (Flurstück minus Gebäude) und
   - optional den **Dachflächen** (eigener Dachflächen-Layer)
   nach denselben akustischen Kriterien (Nachbarschutz, Irrelevanzkriterium) bewertet und farbig eingestuft.  
   Damit sieht man, wo auf einem Flurstück (oder Dach) Aufstellbereiche grundsätzlich zulässig, irrelevant oder problematisch wären.

**Kernparameter**
- *WP-Lärm am Quellgebäude [dB]*: Pegel bei Referenzabstand `r_ref` (z. B. 0,3 m oder 1 m).
- *Zulässiger Lärm an Fassade [dB]*: Grenzwert pro Gebäude.
- *Eigen-Immissionszuschlag [dB]*: Wie viel lauter die eigene WP an der **eigenen** Fassade (am Boden) sein darf.
- *Schallschutzhaube [dB]*: Pauschale Dämpfung am Gerät.
- *Dachflächen-Layer*: zusätzliche Polygone, auf denen WP (bei aktivierter Option) platziert werden dürfen.

**Status-Codierung (Optimalmodus)**
- *irrelevant*: Nachbar-Reserve ≥ Irrelevanzschwelle (z. B. 6 dB).
- *ok*: Nachbarn eingehalten (Reserve ≥ 0 dB), Eigen-Grenze eingehalten oder deaktiviert.
- *self-violation*: nur eigenes Gebäude verletzt (Eigen-Grenze unterschritten), Nachbarn ok.
- *violation*: mindestens ein Nachbar-Gebäude verletzt.

**Status im Rastermodus**
- *irrelevant*: Nachbarn sehr komfortabel eingehalten (≥ Irrelevanzschwelle).
- *ok*: Nachbarn eingehalten (≥ 0 dB).
- *violation*: Nachbarn verletzt (< 0 dB).
""")

# ---------- Helpers / Defaults -----------------------------------------
SUPPORTED_EXTS = {".zip", ".gpkg", ".geojson", ".json"}

@st.cache_data(show_spinner=False)
def read_vector_from_upload(uploaded_file) -> gpd.GeoDataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise ValueError(f"Nicht unterstütztes Format: {suffix}. Bitte .zip, .gpkg oder .geojson/.json verwenden.")
    tmpdir = Path(st.session_state.get("_tmpdir", "uploaded_data"))
    tmpdir.mkdir(exist_ok=True, parents=True)
    temppath = tmpdir / uploaded_file.name
    with open(temppath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    gdf = gpd.read_file(f"zip://{temppath}") if suffix == ".zip" else gpd.read_file(temppath)
    gdf.columns = [str(c) for c in gdf.columns]
    return gdf

def show_gdf_preview(gdf: gpd.GeoDataFrame, label: str):
    st.subheader(label)
    left, right = st.columns([2, 1])
    with left:
        st.write("**Spaltenvorschau (erste 10 Zeilen):**")
        st.dataframe(gdf.drop(columns=[c for c in gdf.columns if c.lower() == "geometry"]).head(10))
    with right:
        st.write("**Info:**")
        st.write(f"Anzahl Zeilen: {len(gdf):,}")
        st.write(f"CRS: {gdf.crs}")
        if gdf.crs is None:
            st.warning("Kein Koordinatenbezugssystem (CRS) erkannt.")
        else:
            try:
                is_projected = not gdf.crs.is_geographic
            except Exception:
                is_projected = None
            if is_projected is False:
                st.warning("CRS ist geographisch (Grad). Für Distanzen bitte metrisches CRS (z. B. EPSG:25832).")
            elif is_projected is True:
                st.success("CRS ist metrisch.")

def numeric_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    return [
        c for c, dt in gdf.dtypes.items()
        if c.lower() != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])
    ]

def id_like_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    candidates = []
    for c in gdf.columns:
        lc = c.lower()
        if any(k in lc for k in ["id", "kenn", "flurst", "flurstk", "fs", "gemarkung", "geb", "build"]):
            candidates.append(c)
    if not candidates:
        candidates = [c for c in gdf.columns if c.lower() != "geometry"]
    return candidates

def evaluate_raster_for_parcel(
    pid,
    pts,                       # Liste von Point-Geometrien auf diesem Flurstück
    b_with_flurstk,            # GeoDataFrame mit allen Gebäuden + Flurstück-ID
    building_id_col: str,
    flurstk_id_col: str,
    neighbors_tree: STRtree,
    neighbor_geoms,
    neighbor_ids,
    neighbor_parc_ids,
    allowed_db_list,
    geom_wkb_to_ix: dict,
    r_ref: float,
    min_r: float,
    safety_db: float,
    use_self_limit: bool,
    self_extra_db: float,
    use_irrelevance: bool,
    irrelevance_threshold_db: float,
    crs_is_metric: bool,
    src_db_for_parcel: float,
    placement_level: str = "ground",
    use_wall_gain: bool = True,
):
    """
    Bewertet alle Rasterpunkte eines Flurstücks pid.
    - Nachbarn = Gebäude auf anderen Flurstücken
    - eigene Gebäude = Gebäude auf demselben Flurstück (optional Eigen-Grenze)
    Gibt ein GeoDataFrame mit:
      status, reserve_db (Nachbarn), self_margin_db, placement_level,
      worst_neighbor_id/dist, dist_to_own_building_m, nearby_walls, wall_gain_db,
      src_L_eff_db, required_clearance_m, irrelevant
    zurück.
    """
    if not pts:
        return gpd.GeoDataFrame(columns=[
            "geometry", "flurstk_id", "status", "reserve_db", "self_margin_db",
            "placement_level", "worst_neighbor_id", "worst_neighbor_dist",
            "dist_to_own_building_m", "nearby_walls", "wall_gain_db",
            "src_L_eff_db", "required_clearance_m", "irrelevant"
        ], geometry="geometry")

    # Gebäude dieses Flurstücks = "eigene" Gebäude
    own_mask = (b_with_flurstk[flurstk_id_col] == pid)
    own_buildings = b_with_flurstk[own_mask]

    own_geoms = list(own_buildings.geometry.values)

    rows = []

    # worst case Suchradius (+6 dB Wandaufschlag, minus Haube)
    source_db_worst = float(src_db_for_parcel) + 6.0 - float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    r_query = max_relevant_radius_for_building(source_db_worst, allowed_db_list, r_ref, safety_db)

    for p in pts:
        # L1_eff am Punkt (ein fiktives Gerät mit src_db_for_parcel auf diesem Flurstück)
        own_union = union_all([g for g in own_geoms if g is not None]) if own_geoms else None
        L1_eff, n_walls, wall_gain_db = effective_source_db_at_point(
            src_db_for_parcel, p,
            neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
            own_geom=own_union,
            crs_is_metric=crs_is_metric,
            use_wall_gain=use_wall_gain
        )

        # Nachbarn suchen (alle Gebäude, dann split nach gleichem/anderem Flurstück)
        buf = p.buffer(r_query)
        cand_ix = _query_tree_indices(neighbors_tree, buf, geom_wkb_to_ix)

        if not cand_ix:
            buf2 = p.buffer(max(r_query, 50.0))
            cand_ix = _query_tree_indices(neighbors_tree, buf2, geom_wkb_to_ix)

        own_ix = [ix for ix in cand_ix if neighbor_parc_ids[ix] == pid]
        ext_ix = [ix for ix in cand_ix if neighbor_parc_ids[ix] != pid]

        # --- 1) Nachbar-Reserve (ext_ix) ---
        worst_neighbor_margin = np.inf
        worst_neighbor_id = None
        worst_neighbor_dist = None

        if ext_ix:
            for ix in ext_ix:
                d = max(p.distance(neighbor_geoms[ix]), min_r)
                Lp = predict_level_at_distance(L1_eff, d, r_ref)
                allowed_eff = allowed_db_list[ix] - safety_db
                margin = allowed_eff - Lp
                if margin < worst_neighbor_margin:
                    worst_neighbor_margin = margin
                    worst_neighbor_id = neighbor_ids[ix]
                    worst_neighbor_dist = d
        else:
            # keine externen Nachbarn → „sehr große“ Reserve
            worst_neighbor_margin = 60.0

        # --- 2) Eigen-Reserve (own_ix) ---
        self_margin = None
        dist_to_own_min = None
        if use_self_limit and own_ix:
            for ix in own_ix:
                d = max(p.distance(neighbor_geoms[ix]), min_r)
                if dist_to_own_min is None or d < dist_to_own_min:
                    dist_to_own_min = d
                Lp = predict_level_at_distance(L1_eff, d, r_ref)

                allowed_self = allowed_db_list[ix]   # zulässiger Fassadenpegel dieses eigenen Gebäudes
                allowed_self_eff_plus = (allowed_self - safety_db) + self_extra_db

                margin_i = allowed_self_eff_plus - Lp
                if self_margin is None or margin_i < self_margin:
                    self_margin = margin_i
        else:
            if own_ix:
                for ix in own_ix:
                    d = p.distance(neighbor_geoms[ix])
                    if dist_to_own_min is None or d < dist_to_own_min:
                        dist_to_own_min = d

        # --- 3) required_clearance_m (für Info) ---
        req_clearance = required_clearance_radius(
            source_db=L1_eff,
            allowed_db_list=allowed_db_list,
            neighbor_ids=neighbor_ids,
            self_id=None,  # im Rastermodus: rein informativer Wert, ohne „eigenes“ Gebäude
            r_ref=r_ref,
            safety_db=safety_db
        )

        # --- 4) Status bestimmen ---
        if worst_neighbor_margin < 0.0:
            status = "violation"
            irr_flag = False
        else:
            if use_self_limit and (self_margin is not None) and (self_margin < 0.0):
                status = "self-violation"
                irr_flag = False
            else:
                if use_irrelevance and worst_neighbor_margin >= irrelevance_threshold_db:
                    status = "irrelevant"
                    irr_flag = True
                else:
                    status = "ok"
                    irr_flag = False

        rows.append({
            "geometry": p,
            "flurstk_id": pid,
            "status": status,
            "reserve_db": float(worst_neighbor_margin),
            "self_margin_db": None if self_margin is None else float(self_margin),
            "placement_level": placement_level,
            "worst_neighbor_id": worst_neighbor_id,
            "worst_neighbor_dist": None if worst_neighbor_dist is None else float(worst_neighbor_dist),
            "dist_to_own_building_m": None if dist_to_own_min is None else float(dist_to_own_min),
            "nearby_walls": int(n_walls),
            "wall_gain_db": float(wall_gain_db),
            "src_L_eff_db": float(L1_eff),
            "required_clearance_m": float(req_clearance),
            "irrelevant": irr_flag,
        })

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=b_with_flurstk.crs)
    return out


def ensure_state_defaults():
    st.session_state.setdefault("target_epsg", 25832)
    st.session_state.setdefault("flurstks_gdf", None)
    st.session_state.setdefault("buildings_gdf", None)
    st.session_state.setdefault("roofs_gdf", None)          # neuer Dachflächen-Layer
    st.session_state.setdefault("res_gdf", None)            # Ergebnisse optimaler Standort
    st.session_state.setdefault("res_raster_gdf", None)     # Rasterergebnisse
    st.session_state.setdefault("lmax_computed", False)
    st.session_state.setdefault("placement_choice", "Freifeld (keine Wand)")
    st.session_state.setdefault("mode", "Optimaler Standort je Gebäude")

ensure_state_defaults()

# kleines Epsilon zum „immer aussparen“
EPS_EXCLUDE = 0.05  # m

# ---------- No-go-Flächen Boden (Gebäude-Puffer) ------------------------
def build_no_go_by_flurstk(b_with_flurstk: gpd.GeoDataFrame,
                           flurstk_id_col: str,
                           setback_m: float = 0.0) -> Dict[Any, Any]:
    """
    Union aller Gebäude je Flurstück, mit Buffer (>= EPS_EXCLUDE).
    setback_m wird nur als Zusatz-Buffer verwendet; bei 0.0 bleibt nur EPS_EXCLUDE.
    """
    no_go = {}
    buf_base = float(setback_m or 0.0)
    for pid, part in b_with_flurstk.groupby(flurstk_id_col):
        buf = max(buf_base, EPS_EXCLUDE)
        geoms = [g.buffer(buf) for g in part.geometry.values if g is not None]
        if geoms:
            no_go[pid] = union_all(geoms)
    return no_go

# ==================== 1) Datei-Upload ==================================
st.divider()
st.header("1) Daten hochladen")

c1, c2, c3 = st.columns(3)
with c1:
    flurstks_file = st.file_uploader(
        "Flurstücke (.zip, .gpkg, .geojson/.json)",
        type=["zip", "gpkg", "geojson", "json"],
        key="flurstks_file"
    )
with c2:
    buildings_file = st.file_uploader(
        "Gebäude (.zip, .gpkg, .geojson/.json)",
        type=["zip", "gpkg", "geojson", "json"],
        key="buildings_file"
    )
with c3:
    roofs_file = st.file_uploader(
        "Dachflächen (.zip, .gpkg, .geojson/.json, optional)",
        type=["zip", "gpkg", "geojson", "json"],
        key="roofs_file"
    )

flurstks_gdf = st.session_state.get("flurstks_gdf")
buildings_gdf = st.session_state.get("buildings_gdf")
roofs_gdf = st.session_state.get("roofs_gdf")

if flurstks_file:
    try:
        flurstks_gdf = read_vector_from_upload(flurstks_file)
        st.session_state["flurstks_gdf"] = flurstks_gdf
        show_gdf_preview(flurstks_gdf, "Flurstücke – Vorschau")
    except Exception as e:
        st.error(f"Fehler beim Laden der Flurstücke: {e}")

if buildings_file:
    try:
        buildings_gdf = read_vector_from_upload(buildings_file)
        st.session_state["buildings_gdf"] = buildings_gdf
        show_gdf_preview(buildings_gdf, "Gebäude – Vorschau")
    except Exception as e:
        st.error(f"Fehler beim Laden der Gebäude: {e}")

if roofs_file:
    try:
        roofs_gdf = read_vector_from_upload(roofs_file)
        st.session_state["roofs_gdf"] = roofs_gdf
        show_gdf_preview(roofs_gdf, "Dachflächen – Vorschau")
    except Exception as e:
        st.error(f"Fehler beim Laden der Dachflächen: {e}")

# ==================== 2) CRS prüfen & transformieren ===================
st.divider()
st.header("2) CRS prüfen & (optional) transformieren")

col_crs1, _ = st.columns(2)
with col_crs1:
    st.number_input("Ziel-CRS (EPSG)", min_value=1000, max_value=999999,
                    value=st.session_state["target_epsg"], step=1, key="target_epsg")
    st.caption("Empfehlung DE: EPSG:25832 (ETRS89 / UTM 32N)")

def crs_tools(label: str, gdf_key: str):
    gdf = st.session_state.get(gdf_key)
    st.markdown(f"**{label}**")
    if gdf is None:
        st.info("Noch keine Daten geladen.")
        return
    st.write(f"Aktuelles CRS: `{gdf.crs}`")
    col_a, _ = st.columns(2)
    with col_a:
        if gdf.crs is None:
            epsg_assign = st.text_input(f"EPSG für {label} zuweisen", value="", key=f"assign_{gdf_key}")
            if st.button(f"CRS zuweisen ({label})"):
                try:
                    epsg_val = int(epsg_assign.strip())
                    st.session_state[gdf_key] = gdf.set_crs(epsg=epsg_val, allow_override=True)
                    st.success(f"CRS gesetzt: EPSG:{epsg_val}")
                except Exception as e:
                    st.error(f"CRS konnte nicht gesetzt werden: {e}")
        else:
            if st.button(f"Auf Ziel-CRS transformieren ({label})"):
                try:
                    target = int(st.session_state["target_epsg"])
                    st.session_state[gdf_key] = gdf.to_crs(epsg=target)
                    st.success(f"{label} transformiert nach EPSG:{target}.")
                except Exception as e:
                    st.error(f"Fehler bei Transformation: {e}")

col1, col2, col3 = st.columns(3)
with col1:
    crs_tools("Flurstücke", "flurstks_gdf")
with col2:
    crs_tools("Gebäude", "buildings_gdf")
with col3:
    crs_tools("Dachflächen", "roofs_gdf")

flurstks_gdf = st.session_state.get("flurstks_gdf")
buildings_gdf = st.session_state.get("buildings_gdf")
roofs_gdf = st.session_state.get("roofs_gdf")

# ==================== 3) Spaltenauswahl ================================
st.divider()
st.header("3) Spalten auswählen")

# --- Gebäude ---
if buildings_gdf is not None:
    st.subheader("Gebäude")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bld_id_col = st.selectbox(
            "Gebäude-ID / Bezeichner",
            options=id_like_columns(buildings_gdf),
            index=0 if id_like_columns(buildings_gdf) else None,
            key="bld_id_col"
        )
    with col2:
        bld_noise_emitted_col = st.selectbox(
            "WP-Lärm am Quellgebäude [dB] (Quelle)",
            options=numeric_columns(buildings_gdf),
            index=0 if numeric_columns(buildings_gdf) else None,
            key="bld_noise_emitted_col",
            help="Pegel am Referenzabstand r_ref (z. B. 0,3 m oder 1 m)."
        )
    with col3:
        bld_allowed_at_facade_col = st.selectbox(
            "Zulässiger Lärm an Fassade [dB]",
            options=numeric_columns(buildings_gdf),
            index=1 if len(numeric_columns(buildings_gdf)) > 1 else (
                0 if numeric_columns(buildings_gdf) else None
            ),
            key="bld_allowed_at_facade_col"
        )
    with col4:
        bld_hp_power_col = st.selectbox(
            "Wärmepumpen-Leistung [kW] (optional)",
            options=["— keine Auswahl —"] + numeric_columns(buildings_gdf),
            index=0,
            key="bld_hp_power_col"
        )

    col_pow1, col_pow2 = st.columns(2)
    with col_pow1:
        use_power_threshold = st.checkbox(
            "Ausschlussregel: Keine WP, wenn Leistung ≤ Schwelle",
            value=False,
            key="use_power_threshold"
        )
    with col_pow2:
        power_threshold_kw = st.number_input(
            "Leistungsschwelle (kW)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            disabled=not st.session_state.get("use_power_threshold", False),
            key="power_threshold_kw"
        )

# --- Flurstücke ---
if flurstks_gdf is not None:
    st.subheader("Flurstücke")
    parc_id_col = st.selectbox(
        "Flurstück-ID / Bezeichner",
        options=id_like_columns(flurstks_gdf),
        index=0 if id_like_columns(flurstks_gdf) else None,
        key="parc_id_col"
    )

# --- Dachflächen ---
roof_type_col = None
roof_allowed_values: List[str] = []
allow_roof_fallback = False
use_roof_layer = False

if roofs_gdf is not None:
    st.subheader("Dachflächen-Layer")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        use_roof_layer = st.checkbox(
            "Dachflächen-Layer als Aufstellflächen verwenden",
            value=False,
            key="use_roof_layer"
        )
    with col_r2:
        if use_roof_layer:
            roof_type_col = st.selectbox(
                "Spalte mit Dachtyp (z. B. Flachdach, Satteldach)",
                options=["— keine Auswahl —"] + [c for c in roofs_gdf.columns if c.lower() != "geometry"],
                index=0,
                key="roof_type_col"
            )
    with col_r3:
        if use_roof_layer and roof_type_col and roof_type_col != "— keine Auswahl —":
            unique_roofs = (
                roofs_gdf[roof_type_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            unique_roofs = sorted(unique_roofs)
            roof_allowed_values = st.multiselect(
                "Erlaubte Dachtypen (für WP-Aufstellung)",
                options=unique_roofs,
                key="roof_allowed_values"
            )

    if use_roof_layer and roof_type_col and roof_type_col != "— keine Auswahl —" and roof_allowed_values:
        allow_roof_fallback = st.checkbox(
            "Dachflächen als Fallback nutzen, wenn am Boden kein zulässiger Standort gefunden wird (Optimalmodus)",
            value=True,
            key="allow_roof_fallback"
        )
    else:
        allow_roof_fallback = False
        if use_roof_layer:
            st.caption("Bitte Dachtyp-Spalte und mindestens einen Dachtyp auswählen, sonst wird keine Dachaufstellung verwendet.")

# ==================== 4) Eigen-Immission & Modus =======================
st.divider()
st.header("4) Eigen-Immissionsgrenze & Moduswahl")

col_s1, col_s2 = st.columns(2)
with col_s1:
    use_self_limit = st.checkbox(
        "Eigen-Immissionsgrenze für eigene Fassade (Bodenaufstellung) aktivieren",
        value=True,
        key="use_self_limit"
    )
with col_s2:
    self_extra_db = st.number_input(
        "Zulässiger Zusatzpegel am eigenen Gebäude (dB)",
        min_value=0.0,
        value=10.0,
        step=0.5,
        disabled=not st.session_state.get("use_self_limit", True),
        key="self_extra_db"
    )
st.info("Die eigene Wärmepumpe darf an der **eigenen** Fassade (am Boden) den zulässigen Pegel um diesen Betrag überschreiten.")

# --- Moduswahl ---
st.subheader("Berechnungsmodus")
mode = st.radio(
    "Modus wählen",
    options=["Optimaler Standort je Gebäude", "Rasterbewertung aller Zellen"],
    index=0 if st.session_state.get("mode", "Optimaler Standort je Gebäude") == "Optimaler Standort je Gebäude" else 1,
    key="mode"
)

# ==================== 5) Akustik-Parameter =============================
st.divider()
st.header("5) Akustik-Parameter")

colA, colB, colC = st.columns(3)
with colA:
    ref_distance_m = st.number_input(
        "Referenzabstand r_ref (m)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Messabstand des Quellpegels L1 (z. B. 0.0 m, 0.3 m oder 1.0 m).",
        key="ref_distance_m"
    )
with colB:
    min_distance_m = st.number_input(
        "Minimal zu wertender Abstand (m)",
        min_value=0.1,
        value=1.0,
        step=0.1,
        help="Untergrenze zur Stabilisierung (typ. 1 m).",
        key="min_distance_m"
    )
with colC:
    safety_margin_db = st.number_input(
        "Sicherheitsaufschlag (dB)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        help="Wird vom erlaubten Fassadenpegel abgezogen.",
        key="safety_margin_db"
    )

st.subheader("Schallschutzhaube")
enclosure_atten_db = st.number_input(
    "Schallschutzhaube: Dämpfung [dB]",
    min_value=0.0, max_value=40.0, value=0.0, step=1.0,
    help="Wird pauschal vom Quellpegel abgezogen (am Gerät).",
    key="enclosure_atten_db"
)

st.subheader("Irrelevanz-Kriterium / Bewertungsmodus")
col_i1, col_i2 = st.columns([2, 1])
with col_i1:
    use_irrelevance = st.checkbox(
        "Irrelevanzkriterium anwenden",
        value=False,
        key="use_irrelevance",
        help="Wenn aktiv: Es wird priorisiert ein Standort/Rasterpunkt mit Reserve ≥ Schwelle gewählt/markiert."
    )
with col_i2:
    irrelevance_threshold_db = st.number_input(
        "Irrelevanz-Schwelle (dB)",
        min_value=0.0, value=6.0, step=0.5,
        key="irrelevance_threshold_db"
    )

# ---------- Akustik-Konstanten & Hilfsfunktionen -----------------------
D_WALL_MAX = 3.0  # Meter
K_LN = 20.0 / np.log(10.0)
C_ATT_DB = 5.0
SEP_EPS = 0.1

PLACEMENT_GAINS = {
    "Freifeld (keine Wand)": 0.0,
    "An einer Wand (Halbraum, +3 dB)": 3.0,
    "Ecke / zwei Wände (Viertelraum, +6 dB)": 6.0,
    "Nische / drei Wände (Achtelraum, +9 dB)": 9.0,
}

def effective_source_db(raw_db: float) -> float:
    """Nur für Export/Info; die eigentliche Berechnung arbeitet punktbezogen."""
    gain = PLACEMENT_GAINS.get(st.session_state.get("placement_choice"), 0.0)
    att = float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    return float(raw_db) + float(gain) - float(att)

def _query_tree_indices(tree: STRtree, query_geom, geom_wkb_to_ix: dict) -> List[int]:
    if hasattr(tree, "query_bulk"):
        arr = np.array([query_geom], dtype=object)
        hits = tree.query_bulk(arr)
        idx = hits[1] if getattr(hits, "size", 0) else np.array([], dtype=int)
        return np.unique(idx).astype(int).tolist()

    geoms_or_idx = tree.query(query_geom)
    if geoms_or_idx is None:
        items = []
    elif isinstance(geoms_or_idx, list):
        items = geoms_or_idx
    else:
        try:
            items = list(geoms_or_idx)
        except Exception:
            items = [geoms_or_idx]

    if not items:
        return []

    first = items[0]
    if isinstance(first, (int, np.integer)) or (
        isinstance(first, np.ndarray) and np.issubdtype(first.dtype, np.integer)
    ):
        flat = np.array(items).ravel()
        return np.unique(flat).astype(int).tolist()

    out = []
    for g in items:
        ix = geom_wkb_to_ix.get(g.wkb)
        if ix is not None:
            out.append(int(ix))
    return list(dict.fromkeys(out))

# --- Wandlogik ---------------------------------------------------------
def walls_visible_and_gain_at_point(
    p: Point,
    neighbors_tree: STRtree,
    neighbor_geoms: List,
    geom_wkb_to_ix: dict,
    own_geom=None,
    max_dist: float = D_WALL_MAX,
    crs_is_metric: bool = True,
    use_wall_gain: bool = True
):
    if (not crs_is_metric) or (not use_wall_gain):
        return 0, 0.0

    def _visible_to(poly):
        try:
            q = ops.nearest_points(p, poly.boundary)[1]
        except Exception:
            return False
        d = p.distance(q)
        if d > max_dist:
            return False
        seg = LineString([p, q])
        block_ix = _query_tree_indices(neighbors_tree, seg, geom_wkb_to_ix)
        for j in block_ix:
            other = neighbor_geoms[j]
            if other is poly:
                continue
            if seg.crosses(other) or (seg.intersects(other) and not seg.touches(other)):
                return False
        return True

    visible = 0

    if own_geom is not None and _visible_to(own_geom):
        visible += 1

    buf = p.buffer(max_dist)
    cand_ix = _query_tree_indices(neighbors_tree, buf, geom_wkb_to_ix)
    for ix in cand_ix:
        poly = neighbor_geoms[ix]
        if own_geom is not None and poly.equals(own_geom):
            continue
        if _visible_to(poly):
            visible += 1

    gain = 6.0 if visible >= 2 else 3.0 if visible == 1 else 0.0
    return int(visible), float(gain)

def effective_source_db_at_point(
    raw_db: float,
    p: Point,
    neighbors_tree: STRtree,
    neighbor_geoms: List,
    geom_wkb_to_ix: dict,
    own_geom=None,
    crs_is_metric: bool = True,
    use_wall_gain: bool = True
):
    n_walls, wall_gain_db = walls_visible_and_gain_at_point(
        p, neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
        own_geom=own_geom, max_dist=D_WALL_MAX, crs_is_metric=crs_is_metric,
        use_wall_gain=use_wall_gain
    )
    att = float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    L1_eff = float(raw_db) + float(wall_gain_db) - att
    return float(L1_eff), int(n_walls), float(wall_gain_db)

# --- Geometrie & Sampling ----------------------------------------------
def assign_buildings_to_flurstks(buildings_gdf: gpd.GeoDataFrame,
                                 flurstks_gdf: gpd.GeoDataFrame,
                                 building_id_col: str,
                                 flurstk_id_col: str) -> gpd.GeoDataFrame:
    b_centroids = buildings_gdf.copy()
    b_centroids["__centroid"] = b_centroids.geometry.centroid
    join1 = gpd.sjoin(
        b_centroids.set_geometry("__centroid")[[building_id_col, "__centroid"]],
        flurstks_gdf[[flurstk_id_col, "geometry"]],
        how="left",
        predicate="within"
    )
    assigned = join1[[building_id_col, flurstk_id_col]].copy()
    missing_ids = assigned[assigned[flurstk_id_col].isna()][building_id_col].tolist()
    if missing_ids:
        b_miss = buildings_gdf[buildings_gdf[building_id_col].isin(missing_ids)].copy()
        inter = gpd.overlay(
            b_miss[[building_id_col, "geometry"]],
            flurstks_gdf[[flurstk_id_col, "geometry"]],
            how="intersection",
            keep_geom_type=False
        )
        if len(inter):
            inter["__area"] = inter.geometry.area
            idx = inter.groupby(building_id_col)["__area"].idxmax()
            best = inter.loc[idx, [building_id_col, flurstk_id_col]]
            assigned = assigned.merge(best, on=building_id_col, how="left", suffixes=("", "_fb"))
            assigned[flurstk_id_col] = assigned[flurstk_id_col].fillna(assigned[f"{flurstk_id_col}_fb"])
            assigned = assigned[[building_id_col, flurstk_id_col]]
    out = buildings_gdf.merge(assigned, on=building_id_col, how="left")
    return out

def sample_points_in_polygon(poly, step=1.0, add_rep_point=True, max_points=5000) -> List[Point]:
    if poly is None or poly.is_empty:
        return []
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)
    if len(xs) * len(ys) > max_points:
        factor = np.sqrt((len(xs) * len(ys)) / max_points)
        step2 = step * float(np.ceil(factor))
        xs = np.arange(minx, maxx + step2, step2)
        ys = np.arange(miny, maxy + step2, step2)
    pts = [Point(x, y) for x in xs for y in ys]
    prep = prepared.prep(poly)
    pts_in = [p for p in pts if prep.contains(p)]
    if add_rep_point:
        rp = poly.representative_point()
        if rp is not None and prep.contains(rp):
            pts_in.append(rp)
    return pts_in

def prepare_receptors(buildings_gdf: gpd.GeoDataFrame, building_id_col: str):
    neighbor_geoms = list(buildings_gdf.geometry.values)
    neighbor_ids = list(buildings_gdf[building_id_col].values)
    neighbors_tree = STRtree(neighbor_geoms)
    geom_wkb_to_ix = {g.wkb: i for i, g in enumerate(neighbor_geoms)}
    return neighbor_geoms, neighbor_ids, neighbors_tree, geom_wkb_to_ix

# --- Akustische Grundfunktionen ----------------------------------------
def required_clearance_radius(source_db: float,
                              allowed_db_list: List[float],
                              neighbor_ids: List,
                              self_id,
                              r_ref: float,
                              safety_db: float) -> float:
    req = []
    for a, nid in zip(allowed_db_list, neighbor_ids):
        if self_id is not None and nid == self_id:
            continue
        allowed_eff = float(a) - float(safety_db)
        sep_req = np.exp((float(source_db) - allowed_eff - C_ATT_DB) / K_LN)
        sep_req = max(sep_req, SEP_EPS)
        r_req = float(r_ref) + sep_req
        req.append(r_req)
    return float(max(req)) if req else float(r_ref)

def predict_level_at_distance(source_db: float, r2: float, r1: float) -> float:
    sep = max(r2 - r1, SEP_EPS)
    return float(source_db - (K_LN * np.log(sep) + C_ATT_DB))

def max_relevant_radius_for_building(source_db: float,
                                     allowed_list_db: List[float],
                                     r_ref: float,
                                     safety_db: float = 0.0) -> float:
    req = []
    for a in allowed_list_db:
        allowed_eff = a - safety_db
        need = source_db - allowed_eff
        sep_req = np.exp((need - C_ATT_DB) / K_LN)
        r_req = r_ref + sep_req
        req.append(r_req)
    return float(max(req) if req else r_ref)

# ==================== Bewertungsfunktionen (Optimalmodus) ===============
def evaluate_points(
    pts: List[Point],
    raw_source_db: float,
    allowed_db_list: List[float],
    neighbors_tree: STRtree,
    neighbor_geoms: List,
    neighbor_ids: List,
    geom_wkb_to_ix: dict,
    self_building_id,
    r_ref: float,
    min_r: float,
    safety_db: float,
    irr_thresh_db: Optional[float] = None,
    own_building_geom=None,
    crs_is_metric: bool = True,
    use_self_limit: bool = False,
    self_allowed_db: Optional[float] = None,
    self_extra_db: float = 0.0,
    disable_wall_gain: bool = False
):
    """
    Bewertet alle Punkte:
    - neighbor_margin: schlechteste Reserve gegenüber Nachbarn
    - self_margin: Reserve gegenüber eigener Fassade (falls aktiviert)
    Gibt beste Kandidaten zurück (any, irrelevanz, fully_ok).
    """
    if not pts:
        return {
            "best_any": None,
            "best_irrel": None,
            "best_fully_ok": None,
            "has_fully_ok": False,
        }

    source_db_worst = float(raw_source_db) + 6.0 - float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    r_query = max_relevant_radius_for_building(source_db_worst, allowed_db_list, r_ref, safety_db)

    def make_cand(p, neighbor_margin, worst_neigh_id, worst_neigh_dist,
                  dist_to_own, n_walls, wall_gain_db, self_margin, L1_eff):
        return {
            "point": p,
            "neighbor_margin": float(neighbor_margin),
            "worst_neighbor_id": worst_neigh_id,
            "worst_neighbor_dist": None if worst_neigh_dist is None else float(worst_neigh_dist),
            "dist_to_own_building_m": None if dist_to_own is None else float(dist_to_own),
            "n_walls": int(n_walls),
            "wall_gain_db": float(wall_gain_db),
            "self_margin": None if self_margin is None else float(self_margin),
            "src_L_eff_db": float(L1_eff),
        }

    best_any = None
    best_irrel = None
    best_fully_ok = None

    for p in pts:
        L1_eff, n_walls, wall_gain_db = effective_source_db_at_point(
            raw_source_db, p, neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
            own_geom=own_building_geom, crs_is_metric=crs_is_metric,
            use_wall_gain=not disable_wall_gain
        )

        buf = p.buffer(r_query)
        cand_ix = _query_tree_indices(neighbors_tree, buf, geom_wkb_to_ix)
        cand_ix = [ix for ix in cand_ix if neighbor_ids[ix] != self_building_id]

        if not cand_ix:
            buf2 = p.buffer(max(r_query, 50.0))
            cand_ix = _query_tree_indices(neighbors_tree, buf2, geom_wkb_to_ix)
            cand_ix = [ix for ix in cand_ix if neighbor_ids[ix] != self_building_id]

        worst_margin = np.inf
        worst_id = None
        worst_dist = None

        if cand_ix:
            for ix in cand_ix:
                d = max(p.distance(neighbor_geoms[ix]), min_r)
                Lp = predict_level_at_distance(L1_eff, d, r_ref)
                allowed_eff = allowed_db_list[ix] - safety_db
                margin = allowed_eff - Lp
                if margin < worst_margin:
                    worst_margin = margin
                    worst_id = neighbor_ids[ix]
                    worst_dist = d
        else:
            worst_margin = 60.0  # keine Nachbarn -> sehr große Reserve

        dist_to_own = None
        if own_building_geom is not None:
            try:
                dist_to_own = float(p.distance(own_building_geom))
                if dist_to_own < min_r:
                    dist_to_own = min_r
            except Exception:
                dist_to_own = None

        self_margin = None
        if use_self_limit and (self_allowed_db is not None) and (own_building_geom is not None) and (dist_to_own is not None):
            L_own = predict_level_at_distance(L1_eff, dist_to_own, r_ref)
            allowed_self_eff_plus = (float(self_allowed_db) - safety_db) + float(self_extra_db)
            self_margin = allowed_self_eff_plus - L_own

        cand = make_cand(
            p=p,
            neighbor_margin=worst_margin,
            worst_neigh_id=worst_id,
            worst_neigh_dist=worst_dist,
            dist_to_own=dist_to_own,
            n_walls=n_walls,
            wall_gain_db=wall_gain_db,
            self_margin=self_margin,
            L1_eff=L1_eff
        )

        if (best_any is None) or (cand["neighbor_margin"] > best_any["neighbor_margin"]):
            best_any = cand

        if irr_thresh_db is not None and cand["neighbor_margin"] >= irr_thresh_db:
            if (best_irrel is None) or (cand["neighbor_margin"] > best_irrel["neighbor_margin"]):
                best_irrel = cand

        fully_ok = cand["neighbor_margin"] >= 0.0
        if use_self_limit and (self_allowed_db is not None):
            fully_ok = fully_ok and (cand["self_margin"] is not None) and (cand["self_margin"] >= 0.0)

        if fully_ok:
            if (best_fully_ok is None) or (cand["neighbor_margin"] > best_fully_ok["neighbor_margin"]):
                best_fully_ok = cand

    has_fully_ok = best_fully_ok is not None
    return {
        "best_any": best_any,
        "best_irrel": best_irrel,
        "best_fully_ok": best_fully_ok,
        "has_fully_ok": has_fully_ok,
    }

def choose_hp_location_for_building(
    build_row,
    flurstks_by_id: dict,
    building_id_col: str,
    flurstk_id_col: str,
    neighbor_geoms: List,
    neighbor_ids: List,
    neighbors_tree: STRtree,
    geom_wkb_to_ix: dict,
    source_db: float,
    allowed_db_list: List[float],
    self_allowed_db: float,
    r_ref: float,
    min_r: float,
    safety_db: float,
    no_go_by_flurstk: dict,
    crs_is_metric: bool,
    use_self_limit: bool,
    self_extra_db: float,
    roof_candidates_by_flurstk: Optional[dict] = None,
    allow_roof_fallback: bool = False,
) -> dict:
    """
    Sucht Bodenstandort; falls nötig und erlaubt, Dachstandort (auf Dachflächen-Layer)
    """
    flurstk_id = build_row[flurstk_id_col]
    building_geom = build_row.geometry
    if flurstk_id not in flurstks_by_id:
        return {"status": "no-flurstk", "point": None, "reserve_db": None}

    flurstk_geom = flurstks_by_id[flurstk_id]

    no_go = no_go_by_flurstk.get(flurstk_id, None)
    if no_go is not None:
        F_ground = flurstk_geom.difference(no_go)
    else:
        F_ground = flurstk_geom

    grid_step = st.session_state.get("grid_step_m", 1.0)

    pts_ground = sample_points_in_polygon(
        F_ground, step=grid_step, add_rep_point=True, max_points=4000
    ) if (F_ground is not None and not F_ground.is_empty) else []

    irr_on = st.session_state.get("use_irrelevance", False)
    irr_thresh = float(st.session_state.get("irrelevance_threshold_db", 6.0)) if irr_on else None

    ground_eval = None
    if pts_ground:
        ground_eval = evaluate_points(
            pts=pts_ground,
            raw_source_db=source_db,
            allowed_db_list=allowed_db_list,
            neighbors_tree=neighbors_tree,
            neighbor_geoms=neighbor_geoms,
            neighbor_ids=neighbor_ids,
            geom_wkb_to_ix=geom_wkb_to_ix,
            self_building_id=build_row[building_id_col],
            r_ref=r_ref,
            min_r=min_r,
            safety_db=safety_db,
            irr_thresh_db=irr_thresh,
            own_building_geom=building_geom,
            crs_is_metric=crs_is_metric,
            use_self_limit=use_self_limit,
            self_allowed_db=self_allowed_db,
            self_extra_db=self_extra_db,
            disable_wall_gain=False
        )

    def pick_candidate_from_eval(ev):
        if ev is None:
            return None, False, False
        cand_irrel = ev["best_irrel"]
        cand_full = ev["best_fully_ok"]
        cand_any = ev["best_any"]
        has_full = ev["has_fully_ok"]

        irr_thresh_loc = irr_thresh

        cand_full_irrel = None
        if cand_full is not None and irr_thresh_loc is not None and cand_full["neighbor_margin"] >= irr_thresh_loc:
            cand_full_irrel = cand_full

        if irr_on:
            if cand_full_irrel is not None:
                return cand_full_irrel, True, True
            if cand_full is not None:
                return cand_full, True, False
            return cand_any, False, False
        else:
            if cand_full is not None:
                return cand_full, True, False
            return cand_any, False, False

    ground_cand, ground_is_fully_ok, ground_is_irrel = pick_candidate_from_eval(ground_eval)

    # --- Dachflächen-Fallback (mit Dachflächen-Layer) -------------------
    roof_poly = None
    if roof_candidates_by_flurstk is not None:
        roof_geoms = roof_candidates_by_flurstk.get(flurstk_id, [])
        if roof_geoms:
            roof_poly = union_all(roof_geoms) if len(roof_geoms) > 1 else roof_geoms[0]

    try_roof = allow_roof_fallback and (roof_poly is not None) and (not ground_is_fully_ok)

    roof_cand = None
    roof_is_fully_ok = False
    roof_is_irrel = False
    if try_roof:
        pts_roof = sample_points_in_polygon(
            roof_poly, step=grid_step, add_rep_point=True, max_points=4000
        )
        if pts_roof:
            roof_eval = evaluate_points(
                pts=pts_roof,
                raw_source_db=source_db,
                allowed_db_list=allowed_db_list,
                neighbors_tree=neighbors_tree,
                neighbor_geoms=neighbor_geoms,
                neighbor_ids=neighbor_ids,
                geom_wkb_to_ix=geom_wkb_to_ix,
                self_building_id=build_row[building_id_col],
                r_ref=r_ref,
                min_r=min_r,
                safety_db=safety_db,
                irr_thresh_db=irr_thresh,
                own_building_geom=building_geom,
                crs_is_metric=crs_is_metric,
                use_self_limit=False,
                self_allowed_db=None,
                self_extra_db=0.0,
                disable_wall_gain=True
            )
            roof_cand, roof_is_fully_ok, roof_is_irrel = pick_candidate_from_eval(roof_eval)

    placement_level = "ground"
    cand = ground_cand
    is_fully_ok = ground_is_fully_ok
    is_irrel = ground_is_irrel

    if try_roof and roof_cand is not None and roof_cand["neighbor_margin"] >= 0.0:
        placement_level = "roof_surface"
        cand = roof_cand
        is_fully_ok = True
        is_irrel = roof_is_irrel

    if cand is None:
        return {"status": "sampling-failed", "point": None, "reserve_db": None}

    L1_eff_sel = cand["src_L_eff_db"]
    n_walls_sel = cand["n_walls"]
    wall_gain_sel = cand["wall_gain_db"]
    dist_to_own = cand["dist_to_own_building_m"]
    worst_neighbor_id = cand["worst_neighbor_id"]
    worst_neighbor_dist = cand["worst_neighbor_dist"]
    best_reserve = cand["neighbor_margin"]
    self_margin = cand["self_margin"]

    req_clearance = required_clearance_radius(
        source_db=L1_eff_sel,
        allowed_db_list=allowed_db_list,
        neighbor_ids=neighbor_ids,
        self_id=build_row[building_id_col],
        r_ref=r_ref,
        safety_db=safety_db
    )

    irr_on_loc = st.session_state.get("use_irrelevance", False)
    irr_thresh_loc = float(st.session_state.get("irrelevance_threshold_db", 6.0)) if irr_on_loc else None

    if best_reserve is not None and best_reserve < 0.0:
        status = "violation"
        irr_flag = False
    else:
        if placement_level == "ground" and use_self_limit and (self_allowed_db is not None) and (self_margin is not None) and (self_margin < 0.0):
            status = "self-violation"
            irr_flag = False
        else:
            if irr_on_loc and best_reserve is not None and irr_thresh_loc is not None and best_reserve >= irr_thresh_loc:
                status = "irrelevant"
                irr_flag = True
            else:
                status = "ok"
                irr_flag = False

    return {
        "status": status,
        "irrelevant": bool(irr_flag),
        "point": cand["point"],
        "reserve_db": float(best_reserve) if best_reserve is not None else None,
        "worst_neighbor_id": worst_neighbor_id,
        "worst_neighbor_dist": None if worst_neighbor_dist is None else float(worst_neighbor_dist),
        "dist_to_own_building_m": None if dist_to_own is None else float(dist_to_own),
        "required_clearance_m": float(req_clearance),
        "nearby_walls": int(n_walls_sel),
        "wall_gain_db": float(wall_gain_sel),
        "src_L_eff_db": float(L1_eff_sel),
        "self_margin_db": None if self_margin is None else float(self_margin),
        "placement_level": placement_level,
    }

def compute_hp_locations(
    buildings_gdf: gpd.GeoDataFrame,
    flurstks_gdf: gpd.GeoDataFrame,
    roofs_gdf: Optional[gpd.GeoDataFrame],
    building_id_col: str,
    flurstk_id_col: str,
    noise_source_col: str,
    allowed_facade_col: str,
    use_power_threshold: bool,
    power_col: Optional[str],
    power_threshold_kw: float,
    r_ref: float,
    min_r: float,
    safety_db: float,
    use_self_limit: bool,
    self_extra_db: float,
    use_roof_layer: bool,
    roof_type_col: Optional[str],
    roof_allowed_values: List[str],
    allow_roof_fallback: bool,
    max_buildings: Optional[int] = None,
    progress_cb=None
) -> gpd.GeoDataFrame:

    b_with_flurstk = assign_buildings_to_flurstks(buildings_gdf, flurstks_gdf, building_id_col, flurstk_id_col)

    neighbor_geoms, neighbor_ids, neighbors_tree, geom_wkb_to_ix = prepare_receptors(b_with_flurstk, building_id_col)
    allowed_db_list = list(b_with_flurstk[allowed_facade_col].astype(float).values)
    
    try:
        crs_is_metric = not (buildings_gdf.crs.is_geographic if buildings_gdf.crs else True)
    except Exception:
        crs_is_metric = True

    no_go_by_flurstk = build_no_go_by_flurstk(b_with_flurstk, flurstk_id_col, setback_m=0.0)
    flurstks_by_id = {row[flurstk_id_col]: row.geometry for _, row in flurstks_gdf.iterrows()}

    power_col_present = power_col is not None and power_col in b_with_flurstk.columns

    # --- Dachflächen-Kandidaten aus Dachflächen-Layer -------------------
    roof_candidates_by_flurstk: Dict[Any, List[Polygon]] = {}
    if use_roof_layer and allow_roof_fallback and roofs_gdf is not None and roof_type_col and roof_type_col in roofs_gdf.columns and roof_allowed_values:

        # Schritt 1: Dachflächen nach erlaubten Typen filtern
        roof_mask_type = roofs_gdf[roof_type_col].astype(str).isin([str(v) for v in roof_allowed_values])
        roofs_filtered = roofs_gdf[roof_mask_type].copy()

        # Schritt 2: Flurstück-ID an Dachflächen hängen (über centroid within)
        roofs_filtered["__centroid"] = roofs_filtered.geometry.centroid
        join_roof = gpd.sjoin(
            roofs_filtered.set_geometry("__centroid")[["__centroid"]],
            flurstks_gdf[[flurstk_id_col, "geometry"]],
            how="left",
            predicate="within"
        )
        roofs_filtered = roofs_filtered.join(join_roof[[flurstk_id_col]], how="left")

        # Schritt 3 entfällt: Für den Fallback brauchen wir nur die Dachgeometrien je Flurstück.
        # Die Zuordnung Flurstück-ID (flurstk_id_col) ist bereits in roofs_filtered enthalten (via join_roof).
        for ridx, r in roofs_filtered.iterrows():
            geom = r.geometry
            if geom is None or (hasattr(geom, "is_empty") and geom.is_empty):
                continue
            pid = r.get(flurstk_id_col)
            if pd.isna(pid):
                continue
            if pid not in roof_candidates_by_flurstk:
                roof_candidates_by_flurstk[pid] = []
            roof_candidates_by_flurstk[pid].append(geom)

    # --- Leistungs-Schwelle Gebäude für HP --------------------------------
    if use_power_threshold and power_col_present:
        mask_hp = b_with_flurstk[power_col] > power_threshold_kw
    else:
        mask_hp = pd.Series(True, index=b_with_flurstk.index)

    total_rows = len(b_with_flurstk)
    if max_buildings is not None and max_buildings > 0:
        total_rows = min(total_rows, int(max_buildings))

    results = []
    count = 0
    for idx, row in b_with_flurstk.iterrows():
        if max_buildings is not None and max_buildings > 0 and count >= max_buildings:
            break

        if not mask_hp.loc[idx]:
            results.append({
                "building_id": row[building_id_col],
                "flurstk_id": row[flurstk_id_col],
                "status": "no-hp-due-to-threshold",
                "point": None,
                "reserve_db": None,
                "worst_neighbor_id": None,
                "worst_neighbor_dist": None,
                "dist_to_own_building_m": None,
                "required_clearance_m": None,
                "nearby_walls": None,
                "wall_gain_db": None,
                "src_L_eff_db": None,
                "self_margin_db": None,
                "placement_level": None,
                "irrelevant": False,
            })
        else:
            raw_source_db = float(row[noise_source_col])
            self_allowed_db = float(row[allowed_facade_col])

            res = choose_hp_location_for_building(
                build_row=row,
                flurstks_by_id=flurstks_by_id,
                building_id_col=building_id_col,
                flurstk_id_col=flurstk_id_col,
                neighbor_geoms=neighbor_geoms,
                neighbor_ids=neighbor_ids,
                neighbors_tree=neighbors_tree,
                geom_wkb_to_ix=geom_wkb_to_ix,
                source_db=raw_source_db,
                allowed_db_list=allowed_db_list,
                self_allowed_db=self_allowed_db,
                r_ref=r_ref,
                min_r=min_r,
                safety_db=safety_db,
                no_go_by_flurstk=no_go_by_flurstk,
                crs_is_metric=crs_is_metric,
                use_self_limit=use_self_limit,
                self_extra_db=self_extra_db,
                roof_candidates_by_flurstk=roof_candidates_by_flurstk if (use_roof_layer and allow_roof_fallback) else None,
                allow_roof_fallback=allow_roof_fallback if use_roof_layer else False,
            )
            res.update({
                "building_id": row[building_id_col],
                "flurstk_id": row[flurstk_id_col]
            })
            results.append(res)

        count += 1
        if progress_cb is not None:
            progress_cb(count, total_rows)

    out = gpd.GeoDataFrame(results, geometry=[r["point"] for r in results], crs=flurstks_gdf.crs)
    return out

# ==================== Rastermodus: Bewertung aller Zellen ===============
def evaluate_single_point_for_raster(
    p: Point,
    raw_source_db: float,
    r_ref: float,
    min_r: float,
    safety_db: float,
    neighbors_tree: STRtree,
    neighbor_geoms: List,
    neighbor_ids: List,
    allowed_db_list: List[float],
    geom_wkb_to_ix: dict,
    own_geom=None,
    crs_is_metric: bool = True,
    use_wall_gain: bool = True
) -> dict:
    """
    Bewertet einen einzelnen Rasterpunkt nur bzgl. Nachbarn,
    ohne Eigen-Immissionsgrenze. Gibt Reserve, kritischen Nachbarn, required_clearance etc. zurück.
    """
    L1_eff, n_walls, wall_gain_db = effective_source_db_at_point(
        raw_source_db, p, neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
        own_geom=own_geom, crs_is_metric=crs_is_metric, use_wall_gain=use_wall_gain
    )

    # relevanter Radius (worst case: +6 dB Wandaufschlag)
    source_db_worst = float(raw_source_db) + 6.0 - float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    r_query = max_relevant_radius_for_building(source_db_worst, allowed_db_list, r_ref, safety_db)

    buf = p.buffer(r_query)
    cand_ix = _query_tree_indices(neighbors_tree, buf, geom_wkb_to_ix)

    if not cand_ix:
        buf2 = p.buffer(max(r_query, 50.0))
        cand_ix = _query_tree_indices(neighbors_tree, buf2, geom_wkb_to_ix)

    worst_margin = np.inf
    worst_id = None
    worst_dist = None

    if cand_ix:
        for ix in cand_ix:
            d = max(p.distance(neighbor_geoms[ix]), min_r)
            Lp = predict_level_at_distance(L1_eff, d, r_ref)
            allowed_eff = allowed_db_list[ix] - safety_db
            margin = allowed_eff - Lp
            if margin < worst_margin:
                worst_margin = margin
                worst_id = neighbor_ids[ix]
                worst_dist = d
    else:
        worst_margin = 60.0

    req_clearance = required_clearance_radius(
        source_db=L1_eff,
        allowed_db_list=allowed_db_list,
        neighbor_ids=neighbor_ids,
        self_id=None,
        r_ref=r_ref,
        safety_db=safety_db
    )

    return {
        "neighbor_margin": float(worst_margin),
        "worst_neighbor_id": worst_id,
        "worst_neighbor_dist": None if worst_dist is None else float(worst_dist),
        "required_clearance_m": float(req_clearance),
        "src_L_eff_db": float(L1_eff),
        "nearby_walls": int(n_walls),
        "wall_gain_db": float(wall_gain_db),
    }

def compute_raster_map(
    buildings_gdf: gpd.GeoDataFrame,
    flurstks_gdf: gpd.GeoDataFrame,
    roofs_gdf: Optional[gpd.GeoDataFrame],
    building_id_col: str,
    flurstk_id_col: str,
    noise_source_col: str,
    allowed_facade_col: str,
    r_ref: float,
    min_r: float,
    safety_db: float,
    use_roof_layer: bool,
    roof_type_col: Optional[str],
    roof_allowed_values: List[str],
    use_power_threshold: bool,          
    power_col: Optional[str],           
    power_threshold_kw: float,          
    progress_cb=None
) -> gpd.GeoDataFrame:

    """
    Rasterbewertung:
    - Bodenfläche: Flurstück minus Gebäude-Puffer (EPS_EXCLUDE)
    - optional Dachflächen: Dachflächen-Layer nach Typ gefiltert
    Für jeden Rasterpunkt: Nachbar-Reserve (ohne Eigen-Grenze), Status: irrelevant/ok/violation.
    """
    b_with_flurstk = assign_buildings_to_flurstks(buildings_gdf, flurstks_gdf, building_id_col, flurstk_id_col)

    neighbor_geoms, neighbor_ids, neighbors_tree, geom_wkb_to_ix = prepare_receptors(b_with_flurstk, building_id_col)
    # Für Raster nehmen wir für die Quelle den WP-Lärm des Gebäudes, dessen Flurstück wir prüfen;
    # pragmatisch: Mittelwert oder worst-case. Hier: nehmen wir pro Flurstück die max Lautstärke.
    # Für den Rasterpunkt an sich ist das eine konservative Vereinfachung.
    allowed_db_list = list(b_with_flurstk[allowed_facade_col].astype(float).values)
    # Flurstück-ID pro Gebäude für eigene-vs-Nachbarn-Logik
    neighbor_parc_ids = list(b_with_flurstk[flurstk_id_col].values)

    # --- Leistungs-Schwelle je Gebäude (nur für Raster-Auswahl) --------
    power_col_present = power_col is not None and power_col in b_with_flurstk.columns
    if use_power_threshold and power_col_present:
        mask_hp = b_with_flurstk[power_col] > power_threshold_kw
    else:
        # Wenn kein Schwellenkriterium aktiv: alle Gebäude gelten als "relevant"
        mask_hp = pd.Series(True, index=b_with_flurstk.index)


    try:
        crs_is_metric = not (buildings_gdf.crs.is_geographic if buildings_gdf.crs else True)
    except Exception:
        crs_is_metric = True

    no_go_by_flurstk = build_no_go_by_flurstk(b_with_flurstk, flurstk_id_col, setback_m=0.0)
    flurstks_by_id = {row[flurstk_id_col]: row.geometry for _, row in flurstks_gdf.iterrows()}

    # Quellpegel pro Flurstück (konservativ: max WP-Lärm) **nur für aktive Flurstücke**
    src_by_flurstk: Dict[Any, float] = {}
    active_flurstk: set = set()

    for pid, grp in b_with_flurstk.groupby(flurstk_id_col):
        # Nur Gebäude mit Heizleistung > Schwelle berücksichtigen, falls aktiv
        if use_power_threshold and power_col_present:
            grp_hp = grp[mask_hp.loc[grp.index]]
        else:
            grp_hp = grp

        # Wenn auf diesem Flurstück kein "relevantes" Gebäude steht: nicht aktiv
        if grp_hp.empty:
            continue

        active_flurstk.add(pid)

        # Quellpegel: max WP-Lärm aus den relevanten Gebäuden
        if len(grp_hp[noise_source_col].dropna()):
            src_by_flurstk[pid] = float(grp_hp[noise_source_col].max())
        else:
            # Falls kein Lärmwert: 0 dB, aber Flurstück bleibt trotzdem aktiv
            src_by_flurstk[pid] = 0.0


    # Dachflächen-Regionen pro Flurstück
    roof_region_by_flurstk: Dict[Any, Polygon] = {}
    if use_roof_layer and roofs_gdf is not None and roof_type_col and roof_type_col in roofs_gdf.columns and roof_allowed_values:
        roof_mask_type = roofs_gdf[roof_type_col].astype(str).isin([str(v) for v in roof_allowed_values])
        roofs_filtered = roofs_gdf[roof_mask_type].copy()
        roofs_filtered["__centroid"] = roofs_filtered.geometry.centroid
        join_roof = gpd.sjoin(
            roofs_filtered.set_geometry("__centroid")[["__centroid"]],
            flurstks_gdf[[flurstk_id_col, "geometry"]],
            how="left",
            predicate="within"
        )
        roofs_filtered = roofs_filtered.join(join_roof[[flurstk_id_col]], how="left")
        for pid, grp in roofs_filtered.groupby(flurstk_id_col):
            geoms = [g for g in grp.geometry.values if g is not None and not g.is_empty]
            if geoms:
                roof_region_by_flurstk[pid] = union_all(geoms) if len(geoms) > 1 else geoms[0]

    grid_step = st.session_state.get("grid_step_m", 1.0)
    irr_on = st.session_state.get("use_irrelevance", False)
    irr_thresh = float(st.session_state.get("irrelevance_threshold_db", 6.0)) if irr_on else None

    rows = []
    total_flurstks = len(flurstks_by_id)
    done_fl = 0

    for pid, fl_geom in flurstks_by_id.items():
        done_fl += 1
        if progress_cb is not None:
            progress_cb(done_fl, total_flurstks)

        # Flurstücke überspringen, auf denen ausschließlich Gebäude mit
        # Heizleistung <= Schwelle stehen (oder gar keine Gebäude)
        if pid not in active_flurstk:
            continue

        if fl_geom is None or fl_geom.is_empty:
            continue

        src_db = src_by_flurstk.get(pid, 0.0)

        # Bodenfläche
        no_go = no_go_by_flurstk.get(pid, None)
        if no_go is not None:
            F_ground = fl_geom.difference(no_go)
        else:
            F_ground = fl_geom

        pts_ground = sample_points_in_polygon(
            F_ground, step=grid_step, add_rep_point=False, max_points=8000
        ) if (F_ground is not None and not F_ground.is_empty) else []

        if pts_ground:
            gdf_ground = evaluate_raster_for_parcel(
                pid=pid,
                pts=pts_ground,
                b_with_flurstk=b_with_flurstk,
                building_id_col=building_id_col,
                flurstk_id_col=flurstk_id_col,
                neighbors_tree=neighbors_tree,
                neighbor_geoms=neighbor_geoms,
                neighbor_ids=neighbor_ids,
                neighbor_parc_ids=neighbor_parc_ids,
                allowed_db_list=allowed_db_list,
                geom_wkb_to_ix=geom_wkb_to_ix,
                r_ref=r_ref,
                min_r=min_r,
                safety_db=safety_db,
                use_self_limit=st.session_state.get("use_self_limit", True),
                self_extra_db=st.session_state.get("self_extra_db", 0.0),
                use_irrelevance=irr_on,
                irrelevance_threshold_db=irr_thresh if irr_thresh is not None else 0.0,
                crs_is_metric=crs_is_metric,
                src_db_for_parcel=src_db,
                placement_level="ground",
                use_wall_gain=True
            )
            rows.extend(gdf_ground.to_dict(orient="records"))

        # Dachflächen-Raster
        if pid in roof_region_by_flurstk:
            roof_poly = roof_region_by_flurstk[pid]
            pts_roof = sample_points_in_polygon(
                roof_poly, step=grid_step, add_rep_point=False, max_points=8000
            )
            if pts_roof:
                gdf_roof = evaluate_raster_for_parcel(
                    pid=pid,
                    pts=pts_roof,
                    b_with_flurstk=b_with_flurstk,
                    building_id_col=building_id_col,
                    flurstk_id_col=flurstk_id_col,
                    neighbors_tree=neighbors_tree,
                    neighbor_geoms=neighbor_geoms,
                    neighbor_ids=neighbor_ids,
                    neighbor_parc_ids=neighbor_parc_ids,
                    allowed_db_list=allowed_db_list,
                    geom_wkb_to_ix=geom_wkb_to_ix,
                    r_ref=r_ref,
                    min_r=min_r,
                    safety_db=safety_db,
                    use_self_limit=False,  # auf Dach keine Eigen-Grenze
                    self_extra_db=0.0,
                    use_irrelevance=irr_on,
                    irrelevance_threshold_db=irr_thresh if irr_thresh is not None else 0.0,
                    crs_is_metric=crs_is_metric,
                    src_db_for_parcel=src_db,
                    placement_level="roof_surface",
                    use_wall_gain=False      # auf Dach keine Wandgewinne
                )
                rows.extend(gdf_roof.to_dict(orient="records"))

    if not rows:
        return gpd.GeoDataFrame(columns=[
            "flurstk_id", "placement_level", "status", "reserve_db",
            "worst_neighbor_id", "worst_neighbor_dist", "required_clearance_m",
            "src_L_eff_db", "nearby_walls", "wall_gain_db", "geometry"
        ], geometry="geometry", crs=flurstks_gdf.crs)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=flurstks_gdf.crs)
    return gdf

# ==================== 6) Validierung & Fortfahren ======================
st.divider()
st.header("6) Validierung & Fortfahren")

ready = True
errors = []

if buildings_gdf is None or flurstks_gdf is None:
    ready = False
    errors.append("Bitte sowohl Flurstücke als auch Gebäude hochladen.")

if buildings_gdf is not None:
    for lbl, key in [
        ("Gebäude-ID", "bld_id_col"),
        ("WP-Lärm Quelle [dB]", "bld_noise_emitted_col"),
        ("Zulässiger Lärm an Fassade [dB]", "bld_allowed_at_facade_col"),
    ]:
        val = st.session_state.get(key)
        if not val or val not in buildings_gdf.columns:
            ready = False
            errors.append(f"Spalte für **{lbl}** fehlt oder ist ungültig.")
    if st.session_state.get("use_power_threshold", False):
        power_col_name = st.session_state.get("bld_hp_power_col")
        if not power_col_name or power_col_name == "— keine Auswahl —" or power_col_name not in buildings_gdf.columns:
            ready = False
            errors.append("Ausschlussregel aktiv: Bitte Spalte 'Wärmepumpen-Leistung [kW]' auswählen.")

if flurstks_gdf is not None:
    if not st.session_state.get("parc_id_col") or st.session_state.get("parc_id_col") not in flurstks_gdf.columns:
        ready = False
        errors.append("Spalte für **Flurstück-ID** fehlt oder ist ungültig.")

if buildings_gdf is not None and flurstks_gdf is not None:
    if buildings_gdf.crs is None or flurstks_gdf.crs is None:
        ready = False
        errors.append("Mindestens eines der Datasets hat kein CRS.")
    elif buildings_gdf.crs != flurstks_gdf.crs:
        ready = False
        errors.append("CRS stimmen nicht überein. Bitte gleiche Projektion verwenden.")

for e in errors:
    st.error(e)

proceed = st.button("Anklicken, damit Parameter gespeichert werden", disabled=not ready, use_container_width=True)
if proceed and ready:
    st.success("Einstellungen übernommen. Nächster Schritt: Berechnung (je nach gewähltem Modus).")

# ==================== 7) Berechnung (Optimal / Raster) =================
st.divider()
st.header("7) Berechnung")

res_gdf = None
res_raster_gdf = None

if ready:
    st.subheader("Allgemeine Parameter für Raster / Standortsuche")
    cA, cB = st.columns(2)
    with cA:
        grid_step_m = st.number_input(
            "Rasterweite für Sampling (m)",
            min_value=0.5,
            value=1.0,
            step=0.5,
            key="grid_step_m",
            help="Kleiner = genauer, aber langsamer."
        )
    with cB:
        max_buildings = st.number_input(
            "Max. Anzahl Gebäude (nur für 'Optimaler Standort je Gebäude'; 0 = alle)",
            min_value=0,
            value=0,
            step=100,
            key="max_buildings"
        )

    run = st.button("Berechnung starten", use_container_width=True)
    if run:
        try:
            if buildings_gdf.crs.is_geographic:
                st.warning("Das CRS ist geographisch (Grad). Für Distanzen ist ein metrisches CRS empfohlen.")
        except Exception:
            pass

        mb = None if max_buildings == 0 else int(max_buildings)
        progress = st.progress(0, text="Starte Berechnung …")
        status_box = st.empty()

        def _progress_cb(i, total):
            frac = i / total if total else 1.0
            progress.progress(frac, text=f"Verarbeite {i}/{total} …")

        with st.spinner("HPT3000 bei der Arbeit …"):
            if mode == "Optimaler Standort je Gebäude":
                res_gdf = compute_hp_locations(
                    buildings_gdf=buildings_gdf,
                    flurstks_gdf=flurstks_gdf,
                    roofs_gdf=roofs_gdf,
                    building_id_col=st.session_state["bld_id_col"],
                    flurstk_id_col=st.session_state["parc_id_col"],
                    noise_source_col=st.session_state["bld_noise_emitted_col"],
                    allowed_facade_col=st.session_state["bld_allowed_at_facade_col"],
                    use_power_threshold=st.session_state.get("use_power_threshold", False),
                    power_col=None if st.session_state.get("bld_hp_power_col") in (None, "— keine Auswahl —") else st.session_state["bld_hp_power_col"],
                    power_threshold_kw=st.session_state.get("power_threshold_kw", 0.0),
                    r_ref=ref_distance_m,
                    min_r=min_distance_m,
                    safety_db=safety_margin_db,
                    use_self_limit=st.session_state.get("use_self_limit", True),
                    self_extra_db=st.session_state.get("self_extra_db", 0.0),
                    use_roof_layer=st.session_state.get("use_roof_layer", False),
                    roof_type_col=st.session_state.get("roof_type_col") if st.session_state.get("roof_type_col") not in (None, "— keine Auswahl —") else None,
                    roof_allowed_values=st.session_state.get("roof_allowed_values", []),
                    allow_roof_fallback=st.session_state.get("allow_roof_fallback", False),
                    max_buildings=mb,
                    progress_cb=_progress_cb
                )
                st.session_state["res_gdf"] = res_gdf
                st.session_state["res_raster_gdf"] = None
                st.session_state["lmax_computed"] = False
                status_box.success("Standortsuche mit Lärmprüfung abgeschlossen.")
            else:
                res_raster_gdf = compute_raster_map(
                    buildings_gdf=buildings_gdf,
                    flurstks_gdf=flurstks_gdf,
                    roofs_gdf=roofs_gdf if st.session_state.get("use_roof_layer", False) else None,
                    building_id_col=st.session_state["bld_id_col"],
                    flurstk_id_col=st.session_state["parc_id_col"],
                    noise_source_col=st.session_state["bld_noise_emitted_col"],
                    allowed_facade_col=st.session_state["bld_allowed_at_facade_col"],
                    r_ref=ref_distance_m,
                    min_r=min_distance_m,
                    safety_db=safety_margin_db,
                    use_roof_layer=st.session_state.get("use_roof_layer", False),
                    roof_type_col=st.session_state.get("roof_type_col") if st.session_state.get("roof_type_col") not in (None, "— keine Auswahl —") else None,
                    roof_allowed_values=st.session_state.get("roof_allowed_values", []),
                    use_power_threshold=st.session_state.get("use_power_threshold", False),      # NEU
                    power_col=(
                        None
                        if st.session_state.get("bld_hp_power_col") in (None, "— keine Auswahl —")
                        else st.session_state["bld_hp_power_col"]
                    ),                                                                           # NEU
                    power_threshold_kw=st.session_state.get("power_threshold_kw", 0.0),          # NEU
                    progress_cb=_progress_cb
                )
                st.session_state["res_raster_gdf"] = res_raster_gdf
                st.session_state["res_gdf"] = None
                st.session_state["lmax_computed"] = False
                status_box.success("Rasterbewertung abgeschlossen.")

        progress.progress(1.0, text="Fertig")

        # --- Übersicht Tabellen -----------------------------------------
        if mode == "Optimaler Standort je Gebäude" and res_gdf is not None:
            st.write("**Status-Zusammenfassung (Optimalmodus):**")
            status_counts = res_gdf["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="anzahl")
            st.dataframe(status_counts)

            base_cols = [
                "building_id", "flurstk_id", "status", "reserve_db",
                "worst_neighbor_id", "worst_neighbor_dist",
                "dist_to_own_building_m", "required_clearance_m", "src_L_eff_db",
                "placement_level", "self_margin_db"
            ]
            debug_cols = ["nearby_walls", "wall_gain_db"]
            cols_show = base_cols + debug_cols
            for c in cols_show:
                if c not in res_gdf.columns:
                    res_gdf[c] = None

            st.write("**Ergebnistabelle (erste 20):**")
            preview = res_gdf[cols_show].copy()
            preview["geometry_wkt"] = res_gdf.geometry.apply(lambda g: None if g is None else g.wkt)
            st.dataframe(preview.head(20))

            if "nearby_walls" in res_gdf.columns and "dist_to_own_building_m" in res_gdf.columns:
                diag_col = "diag_ownwall_expected"
                res_gdf[diag_col] = (
                    (pd.to_numeric(res_gdf["dist_to_own_building_m"], errors="coerce") <= 3.0)
                    & (pd.to_numeric(res_gdf["nearby_walls"], errors="coerce") == 0)
                    & (res_gdf["placement_level"] == "ground")
                )
                st.write("**Diagnose (eigene Wand erwartet, aber nicht gezählt – Bodenaufstellung):**")
                st.dataframe(
                    res_gdf.loc[res_gdf[diag_col] == True,  # noqa: E712
                                ["building_id", "flurstk_id", "dist_to_own_building_m",
                                 "nearby_walls", "wall_gain_db", "placement_level"]].head(20)
                )
                st.session_state["res_gdf"] = res_gdf

        elif mode == "Rasterbewertung aller Zellen" and res_raster_gdf is not None:
            st.write("**Status-Zusammenfassung (Rastermodus):**")
            status_counts = res_raster_gdf["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="anzahl")
            st.dataframe(status_counts)

            st.write("**Rasterpunkte (erste 20):**")
            preview = res_raster_gdf[[
                "flurstk_id", "placement_level", "status", "reserve_db",
                "worst_neighbor_id", "worst_neighbor_dist",
                "required_clearance_m", "src_L_eff_db",
                "nearby_walls", "wall_gain_db"
            ]].copy()
            preview["geometry_wkt"] = res_raster_gdf.geometry.apply(lambda g: None if g is None else g.wkt)
            st.dataframe(preview.head(20))

# ==================== Erklärung Ausgabespalten =========================
with st.expander("Erklärung der wichtigsten Ausgabespalten"):
    st.markdown("""
**Optimaler Standort je Gebäude**

- **status**:  
  - *irrelevant* = Reserve zu allen Nachbarn ≥ Irrelevanzschwelle  
  - *ok* = Nachbarn eingehalten (Reserve ≥ 0 dB), Eigen-Grenze eingehalten oder deaktiviert  
  - *self-violation* = nur Eigen-Grenze verletzt, Nachbarn eingehalten  
  - *violation* = mindestens ein Nachbar verletzt
- **reserve_db**: kleinste Reserve gegenüber dem kritischsten Nachbarn (dB).
- **src_L_eff_db**: effektiver Quellpegel am Standort (inkl. Haube und ggf. Wandaufschlag).  
- **self_margin_db**: Reserve zur eigenen Fassade (nur für Bodenaufstellung relevant, wenn Eigen-Grenze aktiv).
- **required_clearance_m**: erforderlicher Mindestabstand der Nachbarfassaden bei gegebenem Standort.
- **nearby_walls / wall_gain_db**: sichtbare Wände ≤ 3 m und resultierender Aufschlag (+3/+6 dB, auf Dachflächen = 0).
- **placement_level**: `"ground"` (Boden) oder `"roof_surface"` (Dachfläche aus Dachflächen-Layer).

**Rasterbewertung**

- **status**:  
  - *irrelevant* = Nachbarn komfortabel eingehalten (Reserve ≥ Irrelevanzschwelle)  
  - *ok* = Nachbarn eingehalten (Reserve ≥ 0 dB)  
  - *violation* = Nachbarn verletzt (Reserve < 0 dB)  
  (Eigen-Grenze wird im Rastermodus **nicht** bewertet.)
- **reserve_db**: Reserve gegenüber dem kritischsten Nachbarn am Rasterpunkt.
- **placement_level**: `"ground"` (Bodenraster) oder `"roof_surface"` (Dachflächenraster).
""")

# ==================== 8) Karte ========================================
st.divider()
st.header("8) Karte")

res_gdf = st.session_state.get("res_gdf")
res_raster_gdf = st.session_state.get("res_raster_gdf")

if (res_gdf is not None and isinstance(res_gdf, gpd.GeoDataFrame) and len(res_gdf)) or \
   (res_raster_gdf is not None and isinstance(res_raster_gdf, gpd.GeoDataFrame) and len(res_raster_gdf)):

    copt1, copt2, copt3 = st.columns(3)
    with copt1:
        show_flurstks = st.checkbox("Flurstücke zeigen", value=True)
    with copt2:
        show_buildings = st.checkbox("Gebäude zeigen", value=False)
    with copt3:
        compute_lmax_all = st.checkbox(
            "Zulässigen Quellpegel (Lmax) je Standort/Rasterpunkt berechnen (nur Optimalmodus)",
            value=False,
            help="Lmax = maximaler Quellpegel, der an allen Nachbarn (inkl. Sicherheitsaufschlag) noch zulässig wäre."
        )

    bld_id_col = st.session_state["bld_id_col"]
    parc_id_col = st.session_state["parc_id_col"]
    noise_col = st.session_state["bld_noise_emitted_col"]
    allow_col = st.session_state["bld_allowed_at_facade_col"]

    neighbor_geoms = list(buildings_gdf.geometry.values)
    neighbor_ids = list(buildings_gdf[bld_id_col].values)
    neighbors_tree = STRtree(neighbor_geoms)
    allowed_db_list = list(buildings_gdf[allow_col].astype(float).values)
    geom_wkb_to_ix = {g.wkb: i for i, g in enumerate(neighbor_geoms)}

    try:
        crs_is_metric_map = not (buildings_gdf.crs.is_geographic if buildings_gdf.crs else True)
    except Exception:
        crs_is_metric_map = True

    def status_color_opt(s):
        s = (s or "").lower()
        if s == "irrelevant":
            return [0, 180, 0, 255]        # grün
        elif s == "ok":
            return [240, 200, 0, 255]      # gelb
        elif s == "self-violation":
            return [255, 165, 0, 255]      # orange
        elif s == "violation":
            return [220, 0, 0, 255]        # rot
        else:
            return [160, 160, 160, 200]

    def status_color_raster(s):
        s = (s or "").lower()
        if s == "irrelevant":
            return [0, 180, 0, 255]        # grün
        elif s == "ok":
            return [240, 200, 0, 255]      # gelb
        elif s == "self-violation":
            return [255, 165, 0, 255]      # orange
        elif s == "violation":
            return [220, 0, 0, 255]        # rot
        else:
            return [160, 160, 160, 200]

    layers = []

    if show_flurstks:
        try:
            flurstks_wgs = flurstks_gdf.to_crs(4326)
            layers.append(pdk.Layer(
                "GeoJsonLayer",
                data=json.loads(flurstks_wgs.to_json()),
                stroked=True,
                filled=False,
                get_line_color=[50, 50, 50, 120],
                get_line_width=1
            ))
        except Exception:
            pass

    if show_buildings:
        try:
            bld_wgs = buildings_gdf.to_crs(4326)
            layers.append(pdk.Layer(
                "GeoJsonLayer",
                data=json.loads(bld_wgs.to_json()),
                stroked=False,
                filled=True,
                get_fill_color=[0, 120, 255, 40]
            ))
        except Exception:
            pass

    # ---------- Optimalmodus: Punkte + Ringe + Lmax ---------------------
    if mode == "Optimaler Standort je Gebäude" and \
       res_gdf is not None and isinstance(res_gdf, gpd.GeoDataFrame) and len(res_gdf):

        results = res_gdf.copy()
        results["color"] = results["status"].apply(status_color_opt)

        bld_attrs = buildings_gdf[[bld_id_col, noise_col, allow_col]].copy()
        bld_attrs.columns = [bld_id_col, "src_L_db", "allowed_facade_db"]
        results = results.merge(bld_attrs, left_on="building_id", right_on=bld_id_col, how="left")

        def max_source_db_at_point(
            p: Point,
            self_building_id,
            neighbors_tree: STRtree,
            neighbor_geoms: List,
            neighbor_ids: List,
            allowed_db_list: List[float],
            geom_wkb_to_ix: dict,
            r_ref: float,
            min_r: float,
            safety_db: float,
            search_radius: float = 500.0
        ) -> float:
            buf = p.buffer(search_radius)
            cand_ix = _query_tree_indices(neighbors_tree, buf, geom_wkb_to_ix)
            cand_ix = [ix for ix in cand_ix if neighbor_ids[ix] != self_building_id]

            if not cand_ix:
                buf2 = p.buffer(max(search_radius, 50.0))
                cand_ix = _query_tree_indices(neighbors_tree, buf2, geom_wkb_to_ix)
                cand_ix = [ix for ix in cand_ix if neighbor_ids[ix] != self_building_id]

            if not cand_ix:
                return float("+inf")
            bounds = []
            for ix in cand_ix:
                d = max(p.distance(neighbor_geoms[ix]), min_r)
                sep = max(d - r_ref, SEP_EPS)
                bound = (float(allowed_db_list[ix]) - safety_db) + (K_LN * np.log(sep) + C_ATT_DB)
                bounds.append(bound)
            return float(min(bounds)) if bounds else float("+inf")

        bld_geom_map = dict(zip(buildings_gdf[bld_id_col].values, buildings_gdf.geometry.values))

        if compute_lmax_all and not st.session_state.get("lmax_computed", False):
            results["Lmax_db"] = np.nan
            results["headroom_db"] = np.nan
            with st.spinner("Berechne Lmax je Standort …"):
                for idx, r in results.iterrows():
                    pt = r.geometry
                    if isinstance(pt, Point):
                        Lmax = max_source_db_at_point(
                            p=pt,
                            self_building_id=r["building_id"],
                            neighbors_tree=neighbors_tree,
                            neighbor_geoms=neighbor_geoms,
                            neighbor_ids=neighbor_ids,
                            allowed_db_list=allowed_db_list,
                            geom_wkb_to_ix=geom_wkb_to_ix,
                            r_ref=ref_distance_m,
                            min_r=min_distance_m,
                            safety_db=safety_margin_db,
                            search_radius=500.0
                        )
                        results.at[idx, "Lmax_db"] = Lmax
                        if pd.notna(r.get("src_L_db")):
                            own_geom_here = bld_geom_map.get(r["building_id"])
                            placement_level = r.get("placement_level", "ground")
                            use_wall_gain_here = (placement_level != "roof_surface")
                            L1_eff_pt, n_w_pt, gain_pt = effective_source_db_at_point(
                                float(r["src_L_db"]), pt,
                                neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
                                own_geom=own_geom_here,
                                crs_is_metric=crs_is_metric_map,
                                use_wall_gain=use_wall_gain_here
                            )
                            results.at[idx, "src_L_eff_db"] = L1_eff_pt
                            if np.isfinite(Lmax):
                                results.at[idx, "headroom_db"] = Lmax - L1_eff_pt
                            if "nearby_walls" not in results.columns or pd.isna(results.at[idx, "nearby_walls"]):
                                results.at[idx, "nearby_walls"] = n_w_pt
                            if "wall_gain_db" not in results.columns or pd.isna(results.at[idx, "wall_gain_db"]):
                                results.at[idx, "wall_gain_db"] = gain_pt
            st.session_state["res_gdf"] = results
            st.session_state["lmax_computed"] = True
        else:
            # Fuck ey HIER ist das!
            results = st.session_state.get("res_gdf", res_gdf).copy()
            if "color" not in results.columns:
                results["color"] = results["status"].apply(status_color_opt)

        pts = results[results.geometry.apply(lambda g: isinstance(g, Point))].copy()
        pts_wgs = None
        if len(pts) > 0:
            pts_wgs = pts.to_crs(4326).copy()
            pts_wgs["lat"] = pts_wgs.geometry.y
            pts_wgs["lon"] = pts_wgs.geometry.x

        # Ringe (Mindestabstand)
        rings_wgs = None
        if len(pts) > 0 and "required_clearance_m" in pts.columns:
            pts_ring = pts[pd.to_numeric(pts["required_clearance_m"], errors="coerce").notna()].copy()
            if len(pts_ring) > 0:
                ring_geoms = []
                for _, rr in pts_ring.iterrows():
                    if isinstance(rr.geometry, Point):
                        try:
                            ring_geoms.append(rr.geometry.buffer(float(rr["required_clearance_m"]), resolution=32))
                        except Exception:
                            ring_geoms.append(None)
                    else:
                        ring_geoms.append(None)
                keep_cols = ["building_id", "flurstk_id", "status", "required_clearance_m"]
                cols_existing = [c for c in keep_cols if c in pts_ring.columns]
                ring_df = pts_ring[cols_existing].copy()
                rings_gdf = gpd.GeoDataFrame(ring_df, geometry=ring_geoms, crs=flurstks_gdf.crs)
                rings_gdf = rings_gdf[rings_gdf.geometry.notna() & (~rings_gdf.geometry.is_empty)]
                if len(rings_gdf) > 0:
                    rings_wgs = rings_gdf.to_crs(4326)

        if rings_wgs is not None and len(rings_wgs):
            layers.append(pdk.Layer(
                "GeoJsonLayer",
                data=json.loads(rings_wgs.to_json()),
                stroked=True,
                filled=False,
                get_line_color=[128, 0, 200, 200],
                get_line_width=0.5
            ))

        if pts_wgs is not None and len(pts_wgs):
            tooltip_html = (
                "<b>Gebäude:</b> {building_id}<br/>"
                "<b>Status:</b> {status}<br/>"
                "<b>Aufstellung:</b> {placement_level}<br/>"
                "<b>Reserve Nachbarn (dB):</b> {reserve_db}<br/>"
                "<b>Reserve eigene Fassade (dB):</b> {self_margin_db}<br/>"
                "<b>Wände ≤3 m:</b> {nearby_walls}<br/>"
                "<b>Wandaufschlag (dB):</b> {wall_gain_db}<br/>"
                "<b>Quellpegel (roh, dB):</b> {src_L_db}<br/>"
                "<b>Quellpegel effektiv (dB):</b> {src_L_eff_db}<br/>"
                "<b>Distanz eigener Baukörper (m):</b> {dist_to_own_building_m}<br/>"
                "<b>Kritischer Nachbar:</b> {worst_neighbor_id}<br/>"
                "<b>Distanz kritischer Nachbar (m):</b> {worst_neighbor_dist}<br/>"
                "<b>Erforderlicher Mindestabstand (m):</b> {required_clearance_m}<br/>"
                + ("<b>Lmax zulässig (dB):</b> {Lmax_db}<br/><b>Spielraum (dB):</b> {headroom_db}<br/>" if "Lmax_db" in pts_wgs.columns else "")
                + "<b>Flurstück ID:</b> {flurstk_id}"
            )

            cols_for_map = [
                "lon", "lat", "color", "flurstk_id", "placement_level", "status",
                "reserve_db", "self_margin_db", "dist_to_own_building_m",
                "nearby_walls", "wall_gain_db",
                "src_L_eff_db", "worst_neighbor_id", "worst_neighbor_dist",
                "required_clearance_m"
            ]
            if "Lmax_db" in pts_wgs.columns:
                cols_for_map.append("Lmax_db")
            if "headroom_db" in pts_wgs.columns:
                cols_for_map.append("headroom_db")

            df_plot = pts_wgs[cols_for_map].copy()
            df_plot = df_plot.replace({np.inf: None, -np.inf: None})
            df_plot = df_plot.where(pd.notnull(df_plot), None)
            if "worst_neighbor_id" in df_plot.columns:
                df_plot["worst_neighbor_id"] = df_plot["worst_neighbor_id"].astype(str)

            def _to_builtin(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, np.generic):
                    return x.item()
                return x

            for c in df_plot.columns:
                df_plot[c] = df_plot[c].map(_to_builtin)

            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_plot,
                get_position=["lon", "lat"],
                get_radius=2.0,
                radius_min_pixels=6,
                radius_max_pixels=10,
                get_fill_color="color",
                pickable=True
            ))

        if pts_wgs is not None and len(pts_wgs):
            center = union_all(pts_wgs.geometry.values).centroid
            vs_lat, vs_lon = center.y, center.x
        else:
            bbox = flurstks_gdf.to_crs(4326).total_bounds
            vs_lon = (bbox[0] + bbox[2]) / 2
            vs_lat = (bbox[1] + bbox[3]) / 2

        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=vs_lat, longitude=vs_lon, zoom=15, bearing=0, pitch=0),
            map_style=None,
            tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "black"}}
        )
        st.pydeck_chart(r)

        st.caption(
            "Punkte (Optimalmodus): "
            "🟢 irrelevant · 🟡 ok · 🟠 self-violation (nur eigene Fassade verletzt) · 🔴 violation (Nachbarn verletzt). "
            "Ringe zeigen den erforderlichen Mindestabstand zu Nachbarn."
        )

    # ---------- Rastermodus: Rasterpunkte -------------------------------
    elif mode == "Rasterbewertung aller Zellen" and \
         res_raster_gdf is not None and isinstance(res_raster_gdf, gpd.GeoDataFrame) and len(res_raster_gdf):

        results = res_raster_gdf.copy()
        results["color"] = results["status"].apply(status_color_raster)

        pts = results[results.geometry.apply(lambda g: isinstance(g, Point))].copy()
        pts_wgs = None
        if len(pts) > 0:
            pts_wgs = pts.to_crs(4326).copy()
            pts_wgs["lat"] = pts_wgs.geometry.y
            pts_wgs["lon"] = pts_wgs.geometry.x

        if pts_wgs is not None and len(pts_wgs):
            tooltip_html = (
                "<b>Flurstück:</b> {flurstk_id}<br/>"
                "<b>Aufstellung:</b> {placement_level}<br/>"
                "<b>Status:</b> {status}<br/>"
                "<b>Reserve Nachbarn (dB):</b> {reserve_db}<br/>"
                "<b>Wände ≤3 m:</b> {nearby_walls}<br/>"
                "<b>Wandaufschlag (dB):</b> {wall_gain_db}<br/>"
                "<b>Quellpegel effektiv (dB):</b> {src_L_eff_db}<br/>"
                "<b>Kritischer Nachbar:</b> {worst_neighbor_id}<br/>"
                "<b>Distanz kritischer Nachbar (m):</b> {worst_neighbor_dist}<br/>"
                "<b>Erforderlicher Mindestabstand (m):</b> {required_clearance_m}"
            )

            cols_for_map = [
                "lon", "lat", "color", "flurstk_id", "placement_level", "status",
                "reserve_db", "nearby_walls", "wall_gain_db",
                "src_L_eff_db", "worst_neighbor_id", "worst_neighbor_dist",
                "required_clearance_m"
            ]
            df_plot = pts_wgs[cols_for_map].copy()
            df_plot = df_plot.replace({np.inf: None, -np.inf: None})
            df_plot = df_plot.where(pd.notnull(df_plot), None)
            if "worst_neighbor_id" in df_plot.columns:
                df_plot["worst_neighbor_id"] = df_plot["worst_neighbor_id"].astype(str)

            def _to_builtin(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, np.generic):
                    return x.item()
                return x

            for c in df_plot.columns:
                df_plot[c] = df_plot[c].map(_to_builtin)

            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_plot,
                get_position=["lon", "lat"],
                get_radius=1.0,
                radius_min_pixels=3,
                radius_max_pixels=6,
                get_fill_color="color",
                pickable=True
            ))

        if pts_wgs is not None and len(pts_wgs):
            center = union_all(pts_wgs.geometry.values).centroid
            vs_lat, vs_lon = center.y, center.x
        else:
            bbox = flurstks_gdf.to_crs(4326).total_bounds
            vs_lon = (bbox[0] + bbox[2]) / 2
            vs_lat = (bbox[1] + bbox[3]) / 2

        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=vs_lat, longitude=vs_lon, zoom=15, bearing=0, pitch=0),
            map_style=None,
            tooltip={"html": tooltip_html, "style": {"backgroundColor": "white", "color": "black"}}
        )
        st.pydeck_chart(r)

        st.caption(
            "Rasterpunkte: "
            "🟢 irrelevant (Reserve ≥ Irrelevanzschwelle) · "
            "🟡 ok (Reserve ≥ 0 dB) · "
            "🔴 violation (Nachbarn verletzt). "
            "Alle Bewertungen beziehen sich hier nur auf den Nachbarschutz, nicht auf die Eigen-Immission."
        )

else:
    st.info("Noch keine Ergebnisse zum Anzeigen. Bitte zuerst eine Berechnung durchführen.")

# ==================== 9) Export als GeoPackage =========================
st.divider()
st.header("9) Export")

res_gdf = st.session_state.get("res_gdf")
res_raster_gdf = st.session_state.get("res_raster_gdf")

if mode == "Optimaler Standort je Gebäude" and isinstance(res_gdf, gpd.GeoDataFrame) and len(res_gdf):

    cexp1, cexp2 = st.columns([2, 1])
    with cexp1:
        gpkg_name = st.text_input(
            "Dateiname (ohne Endung)",
            value="wp_ergebnis_optimal",
            help="Die Endung .gpkg wird automatisch ergänzt."
        )
    with cexp2:
        export_btn = st.button("GeoPackage exportieren (Punkte, Ringe, rote Flurstücke)", use_container_width=True)

    if export_btn:
        try:
            bld_id_col = st.session_state["bld_id_col"]
            parc_id_col = st.session_state["parc_id_col"]
            noise_col = st.session_state["bld_noise_emitted_col"]
            allow_col = st.session_state["bld_allowed_at_facade_col"]
            power_col = st.session_state.get("bld_hp_power_col")
            use_power = st.session_state.get("use_power_threshold", False) and power_col and power_col != "— keine Auswahl —" and power_col in buildings_gdf.columns

            bld_attrs = [bld_id_col, noise_col, allow_col]
            if use_power:
                bld_attrs.append(power_col)
            bld_attr_df = buildings_gdf[bld_attrs].copy()
            rename_cols = [bld_id_col, "src_L_db", "allowed_facade_db"] + (["hp_power_kw"] if use_power else [])
            bld_attr_df.columns = rename_cols

            results = res_gdf.merge(bld_attr_df, left_on="building_id", right_on=bld_id_col, how="left")

            results["param_self_extra_db"] = float(st.session_state.get("self_extra_db", 0.0))
            results["param_r_ref_m"] = float(st.session_state.get("ref_distance_m", 0.0))
            results["param_min_r_m"] = float(st.session_state.get("min_distance_m", 1.0))
            results["param_safety_db"] = float(st.session_state.get("safety_margin_db", 0.0))
            results["crs_epsg"] = flurstks_gdf.crs.to_epsg() if flurstks_gdf.crs else None

            geom_source = []
            _bld_centroids = buildings_gdf[[bld_id_col, "geometry"]].copy()
            _bld_centroids["__centroid"] = _bld_centroids.geometry.centroid
            cent_map = dict(zip(_bld_centroids[bld_id_col].astype(str), _bld_centroids["__centroid"]))

            new_geoms = []
            for _, r in results.iterrows():
                g = r.geometry
                if g is None or (hasattr(g, "is_empty") and g.is_empty):
                    c = cent_map.get(str(r["building_id"]))
                    new_geoms.append(c if c is not None else None)
                    geom_source.append("building_centroid_placeholder")
                else:
                    new_geoms.append(g)
                    geom_source.append("optimized_location")

            pts_gdf = results.drop(columns="geometry")
            pts_gdf = gpd.GeoDataFrame(pts_gdf, geometry=new_geoms, crs=flurstks_gdf.crs)
            pts_gdf["geom_source"] = geom_source

            pts_gdf["placement_choice"] = str(st.session_state.get("placement_choice"))
            pts_gdf["placement_gain_db"] = float(PLACEMENT_GAINS.get(st.session_state.get("placement_choice"), 0.0))
            pts_gdf["enclosure_atten_db"] = float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)

            irr_on = bool(st.session_state.get("use_irrelevance", False))
            irr_thr = float(st.session_state.get("irrelevance_threshold_db", 6.0)) if irr_on else np.nan
            if "irrelevant" not in pts_gdf.columns:
                pts_gdf["irrelevant"] = False
            pts_gdf["irrelevant"] = pts_gdf["irrelevant"].astype(bool).astype(int)
            pts_gdf["irr_threshold_db"] = irr_thr

            keep = [c for c in [
                "building_id", "flurstk_id", "status", "placement_level",
                "reserve_db", "self_margin_db",
                "nearby_walls", "wall_gain_db",
                "worst_neighbor_id", "worst_neighbor_dist",
                "dist_to_own_building_m", "required_clearance_m",
                "src_L_db", "src_L_eff_db", "allowed_facade_db", "hp_power_kw",
                "placement_choice", "placement_gain_db", "enclosure_atten_db",
                "irrelevant", "irr_threshold_db",
                "param_self_extra_db", "param_r_ref_m", "param_min_r_m", "param_safety_db",
                "crs_epsg", "geom_source",
                "Lmax_db", "headroom_db"
            ] if c in pts_gdf.columns]

            pts_out = pts_gdf[keep + ["geometry"]].copy()

            for c in ["building_id", "flurstk_id", "worst_neighbor_id", "status", "geom_source",
                      "placement_choice", "placement_level"]:
                if c in pts_out.columns:
                    pts_out[c] = pts_out[c].astype("string")

            for c in ["reserve_db", "self_margin_db", "worst_neighbor_dist", "dist_to_own_building_m", "required_clearance_m",
                      "src_L_db", "src_L_eff_db", "allowed_facade_db", "hp_power_kw",
                      "placement_gain_db", "enclosure_atten_db",
                      "irr_threshold_db", "nearby_walls", "wall_gain_db",
                      "param_self_extra_db", "param_r_ref_m", "param_min_r_m", "param_safety_db",
                      "crs_epsg", "Lmax_db", "headroom_db"]:
                if c in pts_out.columns:
                    pts_out[c] = pd.to_numeric(pts_out[c], errors="coerce")

            if "irrelevant" in pts_out.columns:
                pts_out["irrelevant"] = pts_out["irrelevant"].astype("Int64")

            outdir = Path(st.session_state.get("_tmpdir", "uploaded_data"))
            outdir.mkdir(parents=True, exist_ok=True)

            fname = (gpkg_name or "wp_ergebnis_optimal").strip() or "wp_ergebnis_optimal"
            gpkg_path = outdir / f"{fname}.gpkg"

            if gpkg_path.exists():
                try:
                    gpkg_path.unlink()
                except Exception:
                    gpkg_path = outdir / f"{fname}_{np.random.randint(1e6)}.gpkg"

            pts_out.to_file(gpkg_path, driver="GPKG", layer="wp_results")

            # Ringe-Layer
            if "required_clearance_m" in pts_gdf.columns:
                ring_rows = pts_gdf[
                    pts_gdf.geometry.notna()
                    & (~pts_gdf.geometry.is_empty)
                    & pd.to_numeric(pts_gdf["required_clearance_m"], errors="coerce").notna()
                ].copy()
                if len(ring_rows):
                    ring_geoms = []
                    for _, rr in ring_rows.iterrows():
                        try:
                            ring_geoms.append(rr.geometry.buffer(float(rr["required_clearance_m"]), resolution=32))
                        except Exception:
                            ring_geoms.append(None)
                    rings_gdf = gpd.GeoDataFrame(
                        ring_rows[["building_id", "flurstk_id", "status", "placement_level", "required_clearance_m", "irrelevant"]]
                        .astype({"building_id": "string", "flurstk_id": "string", "status": "string", "placement_level": "string"}),
                        geometry=ring_geoms,
                        crs=flurstks_gdf.crs
                    )
                    if "irrelevant" in rings_gdf.columns:
                        rings_gdf["irrelevant"] = rings_gdf["irrelevant"].astype(bool).astype(int)
                    rings_gdf = rings_gdf[rings_gdf.geometry.notna() & (~rings_gdf.geometry.is_empty)]
                    if len(rings_gdf):
                        rings_gdf["required_clearance_m"] = pd.to_numeric(rings_gdf["required_clearance_m"], errors="coerce")
                        rings_gdf.to_file(gpkg_path, driver="GPKG", layer="wp_rings")

            # Rote Flurstücke-Layer (keine Lösung)
            mask_no_solution = (
                res_gdf.geometry.isna() |
                res_gdf.geometry.apply(lambda g: not isinstance(g, Point))
            ) & (res_gdf["status"] != "no-hp-due-to-threshold")
            no_sol_ids = set(res_gdf.loc[mask_no_solution, "flurstk_id"].dropna().astype(str).values)

            if len(no_sol_ids) > 0:
                red_flurstks_gdf = flurstks_gdf[flurstks_gdf[parc_id_col].astype(str).isin(no_sol_ids)].copy()
                red_flurstks_gdf["no_solution"] = 1
                red_flurstks_gdf[parc_id_col] = red_flurstks_gdf[parc_id_col].astype("string")
                red_flurstks_gdf.to_file(gpkg_path, driver="GPKG", layer="wp_flurstks_nosolution")

            with open(gpkg_path, "rb") as f:
                data = f.read()
            st.download_button(
                label="GeoPackage herunterladen",
                data=data,
                file_name=gpkg_path.name,
                mime="application/geopackage+sqlite3",
                use_container_width=True
            )
            st.success(f"GeoPackage erstellt: {gpkg_path.name} (Layer: 'wp_results', 'wp_rings', 'wp_flurstks_nosolution').")

        except Exception as e:
            st.error(f"Export fehlgeschlagen: {e}")

elif mode == "Rasterbewertung aller Zellen" and isinstance(res_raster_gdf, gpd.GeoDataFrame) and len(res_raster_gdf):

    cexp1, cexp2 = st.columns([2, 1])
    with cexp1:
        gpkg_name_r = st.text_input(
            "Dateiname (ohne Endung)",
            value="wp_raster",
            help="Die Endung .gpkg wird automatisch ergänzt."
        )
    with cexp2:
        export_btn_r = st.button("GeoPackage exportieren (Rasterpunkte)", use_container_width=True)

    if export_btn_r:
        try:
            parc_id_col = st.session_state["parc_id_col"]
            results = res_raster_gdf.copy()
            results["crs_epsg"] = flurstks_gdf.crs.to_epsg() if flurstks_gdf.crs else None
            results["param_r_ref_m"] = float(st.session_state.get("ref_distance_m", 0.0))
            results["param_min_r_m"] = float(st.session_state.get("min_distance_m", 1.0))
            results["param_safety_db"] = float(st.session_state.get("safety_margin_db", 0.0))
            results["irr_threshold_db"] = float(st.session_state.get("irrelevance_threshold_db", 6.0)) if st.session_state.get("use_irrelevance", False) else np.nan

            keep = [
                "flurstk_id", "placement_level", "status", "reserve_db",
                "self_margin_db", "dist_to_own_building_m",
                "worst_neighbor_id", "worst_neighbor_dist",
                "required_clearance_m", "src_L_eff_db",
                "nearby_walls", "wall_gain_db",
                "crs_epsg", "param_r_ref_m", "param_min_r_m", "param_safety_db", "irr_threshold_db"
            ]
            pts_out = results[keep + ["geometry"]].copy()

            for c in ["flurstk_id", "placement_level", "status", "worst_neighbor_id"]:
                if c in pts_out.columns:
                    pts_out[c] = pts_out[c].astype("string")

            for c in ["reserve_db", "worst_neighbor_dist", "required_clearance_m", "src_L_eff_db",
                      "nearby_walls", "wall_gain_db", "crs_epsg",
                      "param_r_ref_m", "param_min_r_m", "param_safety_db", "irr_threshold_db"]:
                if c in pts_out.columns:
                    pts_out[c] = pd.to_numeric(pts_out[c], errors="coerce")

            outdir = Path(st.session_state.get("_tmpdir", "uploaded_data"))
            outdir.mkdir(parents=True, exist_ok=True)

            fname = (gpkg_name_r or "wp_raster").strip() or "wp_raster"
            gpkg_path = outdir / f"{fname}.gpkg"

            if gpkg_path.exists():
                try:
                    gpkg_path.unlink()
                except Exception:
                    gpkg_path = outdir / f"{fname}_{np.random.randint(1e6)}.gpkg"

            pts_out.to_file(gpkg_path, driver="GPKG", layer="wp_raster")

            with open(gpkg_path, "rb") as f:
                data = f.read()
            st.download_button(
                label="GeoPackage (Raster) herunterladen",
                data=data,
                file_name=gpkg_path.name,
                mime="application/geopackage+sqlite3",
                use_container_width=True
            )
            st.success(f"GeoPackage erstellt: {gpkg_path.name} (Layer: 'wp_raster').")

        except Exception as e:
            st.error(f"Export fehlgeschlagen: {e}")

else:
    st.info("Keine Ergebnisse zum Export vorhanden. Bitte zuerst die entsprechende Berechnung durchführen.")
