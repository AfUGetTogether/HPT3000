import io
import os
from pathlib import Path
from typing import List

import streamlit as st
import geopandas as gpd
import pandas as pd
import json

# Numerik/Geometrie & Akustik
import numpy as np
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from shapely import prepared
from shapely import union_all
from shapely import ops

import shapely, sys, inspect

# ---------- Page config ----------
st.set_page_config(page_title="Heatpump-Positioning-Tool 3000", layout="wide")
st.title("Heatpump-Positioning-Tool 3000")

# ==== Intro + Expertenmodus ===================================================
with st.expander("Was macht dieses Tool? – aufklappen", expanded=False):
    st.markdown("""
**Das HPT3000** ermittelt je Gebäude den besten Aufstellpunkt einer Wärmepumpe auf dem zugehörigen Flurstück, 
so dass an allen umliegenden Gebäudefassaden die zulässigen Pegel eingehalten werden (inkl. Sicherheitsaufschlag).  
Dabei werden:
- die **zulässigen Fassadenpegel** pro Gebäude (deine Eingabespalte),
- der **Quellpegel am Referenzabstand _r_ref_** (deine Eingabespalte),
- der **Mindestabstand zur eigenen Fassade** (Setback) als **Tabuzone**,
- sowie **nahe Wände ≤ 3 m** mit **Sichtlinie** (automatische +3/+6/+9 dB) berücksichtigt.
Das Tool sampelt Punkte innerhalb der zulässigen Fläche, prüft an jedem Punkt die **schlechteste Reserve** (kritischster Nachbar) 
und wählt den **besten**. Optional wird das **Irrelevanzkriterium** (z. B. Reserve ≥ 6 dB) priorisiert.

**Wichtige Eingänge**
- *WP-Lärm am Quellgebäude [dB]*: Pegel bei **_r_ref_** (z. B. 0,3 m oder 1 m).
- *Zulässiger Lärm an Fassade [dB]*: Grenzwert pro Gebäude.
- *Setback*: Mindestabstand zur **eigenen** Fassade (Fläche wird ausgespart).
- *Schallschutzhaube [dB]*: Pauschale Dämpfung am Gerät.

**Wichtige Ausgänge**
- *status*: „ok“, „irrelevant“ (Reserve ≥ Schwelle), „violation“ (Reserve < 0 dB).
- *reserve_db*: geringste Reserve (kritischste Fassade).
- *required_clearance_m*: erforderlicher Mindestabstand, den der Punkt (bei gegebenem Effektivpegel) zu Nachbarn benötigt.
- *Lmax_db* (optional): maximal zulässiger Quellpegel am Punkt; *headroom_db* = Lmax − L1_eff.
    """)


# ---------- Helpers ----------
SUPPORTED_EXTS = {'.zip', '.gpkg', '.geojson', '.json'}

@st.cache_data(show_spinner=False)
def read_vector_from_upload(uploaded_file) -> gpd.GeoDataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise ValueError(f"Nicht unterstütztes Format: {suffix}. Bitte .zip (Shapefile), .gpkg oder .geojson verwenden")
    tmpdir = Path(st.session_state.get("_tmpdir", "uploaded_data"))
    tmpdir.mkdir(exist_ok=True, parents=True)
    temppath = tmpdir / uploaded_file.name
    with open(temppath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    gdf = gpd.read_file(f"zip://{temppath}", engine="pyogrio") if suffix == ".zip" else gpd.read_file(temppath)
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
            st.warning("Kein Koordinatenbezugssystem (CRS) erkannt")
        else:
            try:
                is_projected = not gdf.crs.is_geographic
            except Exception:
                is_projected = None
            if is_projected is False:
                st.warning("CRS ist geographisch (Grad). Für Distanzen bitte metrisches CRS (z. B. EPSG:25832)")
            elif is_projected is True:
                st.success("CRS ist metrisch")

def numeric_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    return [c for c, dt in gdf.dtypes.items() if str(dt) != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])]

def id_like_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    candidates = []
    for c in gdf.columns:
        lc = c.lower()
        if any(k in lc for k in ["id", "kenn", "flurst", "flurstk", "fs", "gemarkung", "geb", "build"]):
            candidates.append(c)
    if not candidates:
        candidates = [c for c in gdf.columns if c.lower() != "geometry"]
    return candidates

def ensure_state_defaults():
    st.session_state.setdefault("target_epsg", 25832)
    st.session_state.setdefault("flurstks_gdf", None)
    st.session_state.setdefault("buildings_gdf", None)
    st.session_state.setdefault("res_gdf", None)  # Ergebnisse persistieren
    st.session_state.setdefault("lmax_computed", False)  # Merker, ob Lmax für alle Punkte berechnet wurde
ensure_state_defaults()

# HINZUFÜGEN: kleines Epsilon zum „immer aussparen“
EPS_EXCLUDE = 0.05  # m – reicht, um die Fassade sicher auszuschließen

def build_no_go_by_flurstk(b_with_flurstk: gpd.GeoDataFrame,
                          flurstk_id_col: str,
                          setback_m: float) -> dict:
    """Union aller Gebäude je Flurstück, mit Buffer (>= EPS_EXCLUDE)."""
    no_go = {}
    buf = max(float(setback_m or 0.0), EPS_EXCLUDE)
    for pid, part in b_with_flurstk.groupby(flurstk_id_col):
        geoms = [g.buffer(buf) for g in part.geometry.values if g is not None]
        if geoms:
            no_go[pid] = union_all(geoms)
    return no_go

# ---------- 1) Datei-Upload ----------
st.divider()
st.header("1) Daten hochladen")
c1, c2 = st.columns(2)
with c1:
    flurstks_file = st.file_uploader("Flurstücke (.zip, .gpkg, .geojson/.json)",
        type=["zip", "gpkg", "geojson", "json"], key="flurstks_file")
with c2:
    buildings_file = st.file_uploader("Gebäude (.zip, .gpkg, .geojson/.json)",
        type=["zip", "gpkg", "geojson", "json"], key="buildings_file")

flurstks_gdf = st.session_state.get("flurstks_gdf")
buildings_gdf = st.session_state.get("buildings_gdf")

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

# ---------- 2) CRS prüfen & transformieren ----------
st.divider()
st.header("2) CRS prüfen & (optional) transformieren")

col_crs1, _ = st.columns(2)
with col_crs1:
    st.number_input("Ziel-CRS (EPSG)", min_value=1000, max_value=999999,
                    value=st.session_state["target_epsg"], step=1, key="target_epsg")
    st.caption("Empfehlung DE: EPSG:25832 (ETRS89 / UTM 32N)")

def crs_tools(label, gdf_key):
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

col1, col2 = st.columns(2)
with col1: crs_tools("Flurstücke", "flurstks_gdf")
with col2: crs_tools("Gebäude", "buildings_gdf")

# ---------- 3) Spaltenauswahl ----------
st.divider()
st.header("3) Spalten auswählen")

if buildings_gdf is not None:
    st.subheader("Gebäude")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bld_id_col = st.selectbox("Gebäude-ID / Bezeichner",
            options=id_like_columns(buildings_gdf),
            index=0 if id_like_columns(buildings_gdf) else None,
            key="bld_id_col")
    with col2:
        bld_noise_emitted_col = st.selectbox("WP-Lärm am Quellgebäude [dB] (Quelle)",
            options=numeric_columns(buildings_gdf),
            index=0 if numeric_columns(buildings_gdf) else None,
            key="bld_noise_emitted_col",
            help="Pegel am Referenzabstand r_ref (z. B. Messung bei 0,3 m oder 1 m).")
    with col3:
        bld_allowed_at_facade_col = st.selectbox("Zulässiger Lärm an Fassade [dB]",
            options=numeric_columns(buildings_gdf),
            index=1 if len(numeric_columns(buildings_gdf))>1 else 0 if numeric_columns(buildings_gdf) else None,
            key="bld_allowed_at_facade_col")
    with col4:
        bld_hp_power_col = st.selectbox("Wärmepumpen-Leistung [kW] (optional)",
            options=["— keine Auswahl —"] + numeric_columns(buildings_gdf),
            index=0, key="bld_hp_power_col")

    col_pow1, col_pow2 = st.columns([1,1])
    with col_pow1:
        use_power_threshold = st.checkbox("Ausschlussregel: Keine WP, wenn Leistung ≤ Schwelle",
                                          value=False, key="use_power_threshold")
    with col_pow2:
        power_threshold_kw = st.number_input("Leistungsschwelle (kW)", min_value=0.0, value=0.0, step=0.1,
                                             disabled=not st.session_state.get("use_power_threshold", False),
                                             key="power_threshold_kw")

if flurstks_gdf is not None:
    st.subheader("Flurstücke")
    parc_id_col = st.selectbox("Flurstück-ID / Bezeichner",
        options=id_like_columns(flurstks_gdf),
        index=0 if id_like_columns(flurstks_gdf) else None,
        key="parc_id_col")

# ---------- 4) Abstandsregel ----------
st.divider()
st.header("4) Abstandsregel der Wärmepumpe zum zugehörigen Gebäude")
col_s1, col_s2 = st.columns(2)
with col_s1:
    use_setback = st.checkbox("Abstand aktivieren", value=False, key="use_setback")
with col_s2:
    setback_m = st.number_input("Mindestabstand (Meter)", min_value=0.0, value=2.0, step=0.5,
                                disabled=not st.session_state.get("use_setback", False), key="setback_m")
st.info("Für korrekte Meter-Angaben sollte das CRS projiziert sein (z. B. EPSG:25832).")

# ---------- 5) Validierung ----------
st.divider()
st.header("5) Validierung & Fortfahren")
ready = True
errors = []

if buildings_gdf is None or flurstks_gdf is None:
    ready = False; errors.append("Bitte sowohl Flurstücke als auch Gebäude hochladen.")

if buildings_gdf is not None:
    for lbl, key in [("Gebäude-ID", "bld_id_col"),
                     ("WP-Lärm Quelle [dB]", "bld_noise_emitted_col"),
                     ("Zulässiger Lärm an Fassade [dB]", "bld_allowed_at_facade_col")]:
        val = st.session_state.get(key)
        if not val or val not in buildings_gdf.columns:
            ready = False; errors.append(f"Spalte für **{lbl}** fehlt oder ist ungültig.")
    if st.session_state.get("use_power_threshold", False):
        power_col = st.session_state.get("bld_hp_power_col")
        if not power_col or power_col == "— keine Auswahl —" or power_col not in buildings_gdf.columns:
            ready = False; errors.append("Ausschlussregel aktiv: Bitte Spalte 'Wärmepumpen-Leistung [kW]' auswählen.")

if flurstks_gdf is not None:
    if not st.session_state.get("parc_id_col") or st.session_state.get("parc_id_col") not in flurstks_gdf.columns:
        ready = False; errors.append("Spalte für **Flurstück-ID** fehlt oder ist ungültig.")

if buildings_gdf is not None and flurstks_gdf is not None:
    if buildings_gdf.crs is None or flurstks_gdf.crs is None:
        ready = False; errors.append("Mindestens eines der Datasets hat kein CRS.")
    elif buildings_gdf.crs != flurstks_gdf.crs:
        ready = False; errors.append("CRS stimmen nicht überein. Bitte gleiche Projektion verwenden.")

for e in errors: st.error(e)
proceed = st.button("Anklicken, damit Parameter gespeichert werden", disabled=not ready, use_container_width=True)
if proceed and ready:
    st.success("Einstellungen übernommen. Nächster Schritt: Platzierung & Lärmprüfung.")


# ================== 6) Platzierung & Lärm-Compliance ===================


st.divider()
st.header("6) Standorte berechnen")

# --- Akustik-Parameter ---
st.subheader("Akustik-Parameter")
colA, colB, colC = st.columns(3)
with colA:
    ref_distance_m = st.number_input(
    "Referenzabstand r_ref (m)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Messabstand des Quellpegels L1 (z. B. 0.0 m = direkt am Gerät, 0.3 m oder 1.0 m)."
    )
with colB:
    min_distance_m = st.number_input("Minimal zu wertender Abstand (m)", min_value=0.1, value=1.0, step=0.1,
                                     help="Untergrenze zur Stabilisierung (typ. 1 m).")
with colC:
    safety_margin_db = st.number_input("Sicherheitsaufschlag (dB)", min_value=0.0, value=0.0, step=0.5,
                                       help="Wird vom erlaubten Fassadenpegel abgezogen.")

# ==== Aufstellparameter ==================
st.subheader("Schallschutz")
enclosure_atten_db = st.number_input(
    "Schallschutzhaube: Dämpfung [dB]",
    min_value=0.0, max_value=40.0, value=0.0, step=1.0,
    help="Wird pauschal vom Quellpegel abgezogen (am Gerät).",
    key="enclosure_atten_db"
)

st.subheader("Irrelevanz-Kriterium / Bewertungsmodus")
col_i1, col_i2 = st.columns([2,1])
with col_i1:
    use_irrelevance = st.checkbox(
        "Irrelevanzkriterium anwenden",
        value=False,
        key="use_irrelevance",
        help="Wenn aktiv: Es wird priorisiert ein Standort mit Reserve ≥ Schwelle (Standard 6 dB) gewählt."
    )
with col_i2:
    irrelevance_threshold_db = st.number_input(
        "Irrelevanz-Schwelle (dB)",
        min_value=0.0, value=6.0, step=0.5,
        key="irrelevance_threshold_db"
    )

# ---------- Kernfunktionen ----------

# Historischer Platzhalter (nicht mehr zur Berechnung verwendet, nur für Export/Info beibehalten)
PLACEMENT_GAINS = {
    "Freifeld (keine Wand)": 0.0,
    "An einer Wand (Halbraum, +3 dB)": 3.0,
    "Ecke / zwei Wände (Viertelraum, +6 dB)": 6.0,
    "Nische / drei Wände (Achtelraum, +9 dB)": 9.0,
}

def effective_source_db(raw_db: float) -> float:
    # NICHT MEHR IN VERWENDUNG FÜR DIE RECHNUNG – nur zur Abwärtskompatibilität.
    gain = PLACEMENT_GAINS.get(st.session_state.get("placement_choice"), 0.0)
    att  = float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    return float(raw_db) + float(gain) - float(att)

# --- NEU: Wandlogik (≤ 3 m mit Sichtkontakt) ---
D_WALL_MAX = 3.0  # Meter
K_LN = 8.6912
C0   = 1.7698
SEP_EPS = 0.1  # minimale wirksame Separationsstrecke (m), um ln(0) zu vermeiden

def walls_visible_and_gain_at_point(
    p: Point,
    neighbors_tree: STRtree,
    neighbor_geoms: list,
    geom_wkb_to_ix: dict,
    own_geom=None,
    max_dist: float = D_WALL_MAX,
    crs_is_metric: bool = True
):
    # Wenn CRS nicht metrisch -> sichere Seite: kein Wandaufschlag
    if not crs_is_metric:
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
        # Sichtlinie darf keine anderen Polygone schneiden
        block_ix = _query_tree_indices(neighbors_tree, seg, geom_wkb_to_ix)
        for j in block_ix:
            other = neighbor_geoms[j]
            # erlaubt: Berührung am Ziel (q); nicht erlaubt: Schnitt unterwegs
            if other is poly:
                continue
            if seg.crosses(other) or (seg.intersects(other) and not seg.touches(other)):
                return False
        return True

    visible = 0

    # 1) Eigenes Gebäude (falls vorhanden) immer testen
    if own_geom is not None and _visible_to(own_geom):
        visible += 1

    # 2) Weitere Gebäude im Umkreis testen
    buf = p.buffer(max_dist)
    cand_ix = _query_tree_indices(neighbors_tree, buf, geom_wkb_to_ix)
    for ix in cand_ix:
        poly = neighbor_geoms[ix]
        # eigenes Gebäude nicht doppelt werten
        if own_geom is not None and poly.equals(own_geom):
            continue
        if _visible_to(poly):
            visible += 1

    gain = 9.0 if visible >= 3 else 6.0 if visible == 2 else 3.0 if visible == 1 else 0.0
    return int(visible), float(gain)


def effective_source_db_at_point(raw_db: float, p: Point,
                                 neighbors_tree: STRtree,
                                 neighbor_geoms: list,
                                 geom_wkb_to_ix: dict,
                                 own_geom=None,
                                 crs_is_metric: bool = True):
    n_walls, wall_gain_db = walls_visible_and_gain_at_point(
        p, neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
        own_geom=own_geom, max_dist=D_WALL_MAX, crs_is_metric=crs_is_metric
    )
    att = float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    L1_eff = float(raw_db) + float(wall_gain_db) - att
    return float(L1_eff), int(n_walls), float(wall_gain_db)


def assign_buildings_to_flurstks(buildings_gdf: gpd.GeoDataFrame,
                                flurstks_gdf: gpd.GeoDataFrame,
                                building_id_col: str,
                                flurstk_id_col: str) -> gpd.GeoDataFrame:
    b_centroids = buildings_gdf.copy()
    b_centroids["__centroid"] = b_centroids.geometry.centroid
    join1 = gpd.sjoin(
        b_centroids.set_geometry("__centroid")[[building_id_col, "__centroid"]],
        flurstks_gdf[[flurstk_id_col, "geometry"]],
        how="left", predicate="within"
    )
    assigned = join1[[building_id_col, flurstk_id_col]].copy()
    missing_ids = assigned[assigned[flurstk_id_col].isna()][building_id_col].tolist()
    if missing_ids:
        b_miss = buildings_gdf[buildings_gdf[building_id_col].isin(missing_ids)].copy()
        inter = gpd.overlay(b_miss[[building_id_col, "geometry"]],
                            flurstks_gdf[[flurstk_id_col, "geometry"]],
                            how="intersection", keep_geom_type=False)
        if len(inter):
            inter["__area"] = inter.geometry.area
            idx = inter.groupby(building_id_col)["__area"].idxmax()
            best = inter.loc[idx, [building_id_col, flurstk_id_col]]
            assigned = assigned.merge(best, on=building_id_col, how="left", suffixes=("", "_fb"))
            assigned[flurstk_id_col] = assigned[flurstk_id_col].fillna(assigned[f"{flurstk_id_col}_fb"])
            assigned = assigned[[building_id_col, flurstk_id_col]]
    out = buildings_gdf.merge(assigned, on=building_id_col, how="left")
    return out

def feasible_region(flurstk_geom, building_geom, setback_m: float):
    if setback_m and setback_m > 0:
        F = flurstk_geom.difference(building_geom.buffer(setback_m))
    else:
        F = flurstk_geom
    if F.is_empty or F.area < 0.01:
        return None
    return F

def sample_points_in_polygon(poly, step=1.0, add_rep_point=True, max_points=5000):
    if poly.is_empty: return []
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
    neighbor_ids   = list(buildings_gdf[building_id_col].values)
    neighbors_tree = STRtree(neighbor_geoms)
    geom_wkb_to_ix = {g.wkb: i for i, g in enumerate(neighbor_geoms)}
    return neighbor_geoms, neighbor_ids, neighbors_tree, geom_wkb_to_ix

def _query_tree_indices(tree: STRtree, query_geom, geom_wkb_to_ix: dict) -> list[int]:
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
    if isinstance(first, (int, np.integer)) or (isinstance(first, np.ndarray) and np.issubdtype(first.dtype, np.integer)):
        flat = np.array(items).ravel()
        return np.unique(flat).astype(int).tolist()

    out = []
    for g in items:
        ix = geom_wkb_to_ix.get(g.wkb)
        if ix is not None:
            out.append(int(ix))
    return list(dict.fromkeys(out))

def required_clearance_radius(source_db: float,
                              allowed_db_list: List[float],
                              neighbor_ids: List,
                              self_id,
                              r_ref: float,
                              safety_db: float) -> float:
    req = []
    for a, nid in zip(allowed_db_list, neighbor_ids):
        if nid == self_id:
            continue
        allowed_eff = float(a) - float(safety_db)
        sep_req = np.exp((float(source_db) - allowed_eff - C0) / K_LN)
        sep_req = max(sep_req, SEP_EPS)
        r_req = float(r_ref) + sep_req
        req.append(r_req)
    return float(max(req)) if req else float(r_ref)

def predict_level_at_distance(source_db: float, r2: float, r1: float) -> float:
    sep = max(r2 - r1, SEP_EPS)
    return float(source_db - (K_LN * np.log(sep) + C0))

def max_relevant_radius_for_building(source_db: float, allowed_list_db: List[float], r_ref: float, safety_db: float = 0.0) -> float:
    req = []
    for a in allowed_list_db:
        allowed_eff = a - safety_db
        need = source_db - allowed_eff
        sep_req = np.exp((need - C0) / K_LN)
        r_req = r_ref + sep_req
        req.append(r_req)
    return float(max(req) if req else r_ref)

def best_point_by_acoustic_compliance(
    pts,
    raw_source_db: float,                 # roh
    allowed_db_list,
    neighbors_tree: STRtree,
    neighbor_geoms,
    neighbor_ids,
    geom_wkb_to_ix: dict,
    self_building_id,
    r_ref: float,
    min_r: float,
    safety_db: float,
    own_building_geom=None,
    crs_is_metric: bool = True 
):
    if not pts:
        return None, -np.inf, None, None, None, 0, 0.0

    # worst case für Suchradius (max. +9 dB)
    source_db_worst = float(raw_source_db) + 9.0 - float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    r_query = max_relevant_radius_for_building(source_db_worst, allowed_db_list, r_ref, safety_db)

    best = (None, -np.inf, None, None, None, 0, 0.0)

    for p in pts:
        L1_eff, n_walls, wall_gain_db = effective_source_db_at_point(
            raw_source_db, p, neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
            own_geom=own_building_geom, crs_is_metric=crs_is_metric
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
            worst_margin = 60.0

        dist_to_own = None
        if own_building_geom is not None:
            try:
                dist_to_own = float(p.distance(own_building_geom))
            except Exception:
                dist_to_own = None

        cand = (p, float(worst_margin), worst_id, None if worst_dist is None else float(worst_dist),
                dist_to_own, int(n_walls), float(wall_gain_db))

        if cand[1] > best[1]:
            best = cand

    return best

def best_point_with_irrelevance_priority(
    pts,
    raw_source_db: float,
    allowed_db_list,
    neighbors_tree: STRtree,
    neighbor_geoms,
    neighbor_ids,
    geom_wkb_to_ix: dict,
    self_building_id,
    r_ref: float,
    min_r: float,
    safety_db: float,
    irr_thresh_db: float,
    own_building_geom=None,
    crs_is_metric: bool = True
):
    if not pts:
        return None, -np.inf, None, None, None, 0, 0.0, False

    source_db_worst = float(raw_source_db) + 9.0 - float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)
    r_query = max_relevant_radius_for_building(source_db_worst, allowed_db_list, r_ref, safety_db)

    best_any = (None, -np.inf, None, None, None, 0, 0.0)
    best_irrel = (None, -np.inf, None, None, None, 0, 0.0)

    for p in pts:
        L1_eff, n_walls, wall_gain_db = effective_source_db_at_point(
            raw_source_db, p, neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
            own_geom=own_building_geom, crs_is_metric=crs_is_metric
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
            worst_margin = 60.0

        dist_to_own = None
        if own_building_geom is not None:
            try:
                dist_to_own = float(p.distance(own_building_geom))
            except Exception:
                dist_to_own = None

        cand = (p, float(worst_margin), worst_id, None if worst_dist is None else float(worst_dist),
                dist_to_own, int(n_walls), float(wall_gain_db))

        if cand[1] > best_any[1]:
            best_any = cand
        if cand[1] >= irr_thresh_db and cand[1] > best_irrel[1]:
            best_irrel = cand

    if best_irrel[0] is not None:
        return (*best_irrel, True)
    else:
        return (*best_any, False)

def choose_hp_location_for_building(
    build_row,
    flurstks_by_id: dict,
    building_id_col: str,
    flurstk_id_col: str,
    setback_m: float,
    neighbor_geoms,
    neighbor_ids,
    neighbors_tree,
    geom_wkb_to_ix: dict,
    source_db: float,              
    allowed_db_list,
    r_ref: float,
    min_r: float,
    safety_db: float,
    no_go_by_flurstk: dict ,
    crs_is_metric: bool           
):
    flurstk_id = build_row[flurstk_id_col]
    building_geom = build_row.geometry
    if flurstk_id not in flurstks_by_id:
        return {"status": "no-flurstk", "point": None, "reserve_db": None}

    flurstk_geom = flurstks_by_id[flurstk_id]
    no_go = no_go_by_flurstk.get(flurstk_id, None)
    if no_go is not None:
        F = flurstk_geom.difference(no_go)
    else:
        # Fallback (sollte selten nötig sein)
        F = feasible_region(flurstk_geom, building_geom, setback_m)

    if F is None or F.is_empty or F.area < 0.01:
        return {"status": "no-feasible-area", "point": None, "reserve_db": None}


    # --- Sampling der Kandidatenpunkte ---
    pts = sample_points_in_polygon(F, step=st.session_state.get("grid_step_m", 1.0),
                                   add_rep_point=True, max_points=4000)

    # --- Beste Position ermitteln (mit/ohne Irrelevanz-Priorität) ---
    if st.session_state.get("use_irrelevance", False):
        irr_thresh = float(st.session_state.get("irrelevance_threshold_db", 6.0))
        best_p, best_reserve, worst_neigh_id, worst_neigh_dist, dist_to_own, n_walls, wall_gain_db, met_irrel = best_point_with_irrelevance_priority(
            pts=pts,
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
            crs_is_metric=crs_is_metric
        )
    else:
        best_p, best_reserve, worst_neigh_id, worst_neigh_dist, dist_to_own, n_walls, wall_gain_db = best_point_by_acoustic_compliance(
            pts=pts,
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
            own_building_geom=building_geom,
            crs_is_metric=crs_is_metric
        )
        met_irrel = False

    if best_p is None:
        return {"status": "sampling-failed", "point": None, "reserve_db": None}

    # L1_eff am gewählten Punkt (für required_clearance & Export)
    L1_eff_sel, n_walls_sel, wall_gain_sel = effective_source_db_at_point(
        source_db, best_p, neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
        own_geom=building_geom, crs_is_metric=crs_is_metric
    )

    # --- Erforderlicher Mindestabstand mit L1_eff am gewählten Punkt ---
    req_clearance = required_clearance_radius(
        source_db=L1_eff_sel,
        allowed_db_list=allowed_db_list,
        neighbor_ids=neighbor_ids,
        self_id=build_row[building_id_col],
        r_ref=r_ref,
        safety_db=safety_db
    )

    # --- Status-Logik ---
    irr_on = st.session_state.get("use_irrelevance", False)
    irr_thresh = float(st.session_state.get("irrelevance_threshold_db", 6.0)) if irr_on else None

    if irr_on:
        if best_reserve is not None and best_reserve >= irr_thresh:
            status = "irrelevant"
            irr_flag = True
        elif best_reserve is not None and best_reserve >= 0.0:
            status = "ok"
            irr_flag = False
        else:
            status = "violation"
            irr_flag = False
    else:
        status = "ok" if (best_reserve is not None and best_reserve >= 0.0) else "violation"
        irr_flag = False

    return {
        "status": status,
        "irrelevant": bool(irr_flag),
        "point": best_p,
        "reserve_db": float(best_reserve),
        "worst_neighbor_id": worst_neigh_id,
        "worst_neighbor_dist": None if worst_neigh_dist is None else float(worst_neigh_dist),
        "dist_to_own_building_m": None if dist_to_own is None else float(dist_to_own),
        "required_clearance_m": float(req_clearance),
        "nearby_walls": int(n_walls_sel),
        "wall_gain_db": float(wall_gain_sel),
        "src_L_eff_db": float(L1_eff_sel)
    }

def compute_hp_locations(buildings_gdf: gpd.GeoDataFrame,
                         flurstks_gdf: gpd.GeoDataFrame,
                         building_id_col: str, flurstk_id_col: str,
                         noise_source_col: str, allowed_facade_col: str,
                         use_power_threshold: bool, power_col: str|None, power_threshold_kw: float,
                         setback_m: float,
                         r_ref: float, min_r: float, safety_db: float,
                         max_buildings: int|None = None,
                         progress_cb=None):
    b_with_flurstk = assign_buildings_to_flurstks(buildings_gdf, flurstks_gdf, building_id_col, flurstk_id_col)

    # Nachbarstrukturen
    neighbor_geoms, neighbor_ids, neighbors_tree, geom_wkb_to_ix = prepare_receptors(b_with_flurstk, building_id_col)
    allowed_db_list = list(b_with_flurstk[allowed_facade_col].astype(float).values)

    # CRS-Flag für Wandlogik
    try:
        crs_is_metric = not (buildings_gdf.crs.is_geographic if buildings_gdf.crs else True)
    except Exception:
        crs_is_metric = True

    no_go_by_flurstk = build_no_go_by_flurstk(b_with_flurstk, flurstk_id_col, setback_m)
    flurstks_by_id = {row[flurstk_id_col]: row.geometry for _, row in flurstks_gdf.iterrows()}

    if use_power_threshold and power_col and power_col in b_with_flurstk.columns:
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
                "dist_to_own_building_m": None
            })
        else:
            raw_source_db = float(row[noise_source_col])
            # Rohwert reinreichen – Wandaufschlag erfolgt punktbezogen
            res = choose_hp_location_for_building(
                build_row=row,
                flurstks_by_id=flurstks_by_id,
                building_id_col=building_id_col,
                flurstk_id_col=flurstk_id_col,
                setback_m=setback_m,
                neighbor_geoms=neighbor_geoms,
                neighbor_ids=neighbor_ids,
                neighbors_tree=neighbors_tree,
                geom_wkb_to_ix=geom_wkb_to_ix,
                source_db=raw_source_db,
                allowed_db_list=allowed_db_list,
                r_ref=r_ref, min_r=min_r, safety_db=safety_db,
                no_go_by_flurstk=no_go_by_flurstk,
                crs_is_metric=crs_is_metric 
            )
            res.update({"building_id": row[building_id_col], "flurstk_id": row[flurstk_id_col]})
            results.append(res)

        count += 1
        if progress_cb is not None:
            progress_cb(count, total_rows)

    out = gpd.GeoDataFrame(results, geometry=[r["point"] for r in results], crs=flurstks_gdf.crs)
    return out

# ---------- UI: Parameter & Start ----------
if ready:
    st.subheader("Parameter für die Standortsuche")
    cA, cB = st.columns(2)
    with cA:
        grid_step_m = st.number_input("Rasterweite für Sampling (m)",
                                    min_value=0.5, value=1.0, step=0.5, key="grid_step_m",
                                    help="Kleiner = genauer, aber langsamer.")
    with cB:
        max_buildings = st.number_input("Max. Anzahl Gebäude (optional, 0 = alle)",
                                        min_value=0, value=0, step=100)


    run = st.button("Standorte berechnen", use_container_width=True)
    if run:
        try:
            if buildings_gdf.crs.is_geographic:
                st.warning("Das CRS ist geographisch (Grad). Für Distanzen ist ein metrisches CRS empfohlen.")
        except Exception:
            pass

        mb = None if max_buildings == 0 else int(max_buildings)

        progress = st.progress(0, text="Starte Standortsuche …")
        status_box = st.empty()

        def _progress_cb(i, total):
            frac = i / total if total else 1.0
            progress.progress(frac, text=f"Verarbeite Gebäude {i}/{total} …")

        with st.spinner("HPT3000 bei der Arbeit …"):
            res_gdf = compute_hp_locations(
                buildings_gdf=buildings_gdf,
                flurstks_gdf=flurstks_gdf,
                building_id_col=st.session_state["bld_id_col"],
                flurstk_id_col=st.session_state["parc_id_col"],
                noise_source_col=st.session_state["bld_noise_emitted_col"],
                allowed_facade_col=st.session_state["bld_allowed_at_facade_col"],
                use_power_threshold=st.session_state.get("use_power_threshold", False),
                power_col=None if st.session_state.get("bld_hp_power_col") in (None, "— keine Auswahl —") else st.session_state["bld_hp_power_col"],
                power_threshold_kw=st.session_state.get("power_threshold_kw", 0.0),
                setback_m=st.session_state.get("setback_m", 0.0) if st.session_state.get("use_setback", False) else 0.0,
                r_ref=ref_distance_m, min_r=min_distance_m, safety_db=safety_margin_db,
                max_buildings=mb,
                progress_cb=_progress_cb
            )

        progress.progress(1.0, text="Fertig")
        status_box.success("Standortsuche mit Lärmprüfung abgeschlossen.")

        # Ergebnisse speichern
        st.session_state["res_gdf"] = res_gdf
        st.session_state["lmax_computed"] = False

        # Übersicht
        st.write("**Status-Zusammenfassung:**")
        status_counts = res_gdf["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="anzahl")
        st.dataframe(status_counts)

        # --- Vorschau-Tabelle erweitern um Wandfelder ---
        base_cols = ["building_id", "flurstk_id", "status", "reserve_db",
                    "worst_neighbor_id", "worst_neighbor_dist",
                    "dist_to_own_building_m", "required_clearance_m", "src_L_eff_db"]

        debug_cols = ["nearby_walls", "wall_gain_db"]  # nur im Expertenmodus

        cols_show = base_cols + (debug_cols)

        for c in cols_show:
            if c not in res_gdf.columns:
                res_gdf[c] = None

        st.write("**Ergebnistabelle (erste 20):**")
        preview = res_gdf[cols_show].copy()
        preview["geometry_wkt"] = res_gdf.geometry.apply(lambda g: None if g is None else g.wkt)
        st.dataframe(preview.head(20))

        # Optionale Diagnose: Fälle, in denen man aufgrund <=3 m eigene Wand erwarten würde, aber 0 Wände gezählt sind
        if isinstance(res_gdf, gpd.GeoDataFrame) and len(res_gdf):
            if "nearby_walls" in res_gdf.columns and "dist_to_own_building_m" in res_gdf.columns:
                diag_col = "diag_ownwall_expected"
                res_gdf[diag_col] = (
                    (pd.to_numeric(res_gdf["dist_to_own_building_m"], errors="coerce") <= 3.0)
                    & (pd.to_numeric(res_gdf["nearby_walls"], errors="coerce") == 0)
                )
                st.write("**Diagnose (eigene Wand erwartet, aber nicht gezählt):**")
                st.dataframe(res_gdf.loc[res_gdf[diag_col] == True,  # noqa: E712
                                        ["building_id","flurstk_id","dist_to_own_building_m","nearby_walls","wall_gain_db"]].head(20))
                # Ergebnis zurückspeichern
                st.session_state["res_gdf"] = res_gdf

        st.caption("Grün = zulässig, Gelb = beste (aber verletzende) Position. "
                   "Rot eingefärbte Flurstücke: keine Lösung gefunden (außer wenn Gebäude wegen Leistungsschwelle ausgeschlossen wurde). "
                   "Zusatz: Distanz zum kritischsten Nachbarn & zum eigenen Gebäude.")
        
with st.expander("Erklärung der wichtigsten Ausgabespalten"):
    st.markdown("""
- **status**:  
  - *ok* = zulässig (Reserve ≥ 0 dB)  
  - *irrelevant* = Reserve ≥ eingestellter Schwelle (z. B. 6 dB)  
  - *violation* = unzulässig (Reserve < 0 dB)
- **reserve_db**: Schlechteste Reserve gegenüber dem kritischsten Nachbarn (dB). Positiv = bestanden
- **src_L_eff_db**: Effektiver Quellpegel am Punkt: L1@r_ref − Haube + Wandaufschlag (+3/+6/+9 dB bei ≤3 m mit Sichtlinie)
- **required_clearance_m**: Erforderlicher Mindestabstand zu Nachbarfassaden bei diesem Punkt
- **worst_neighbor_id / worst_neighbor_dist**: ID und Distanz des kritischsten Nachbarn (der die Reserve bestimmt)
- **dist_to_own_building_m**: Abstand zur **eigenen** Fassade (Kontrolle Setback)
- **Lmax_db**: Maximal zulässiger Quellpegel am Punkt (unter allen Nachbargrenzwerten)  
  **headroom_db** = Lmax − src_L_eff_db
- **nearby_walls / wall_gain_db**: Anzahl sichtbarer Wände ≤ 3 m und resultierender Aufschlag (+3/+6/+9 dB)
""")


# =============== 7) Karte – alle WP-Standorte (grün/gelb) ==============

st.divider()
st.header("7) Karte")

import pydeck as pdk

res_gdf = st.session_state.get("res_gdf")
if isinstance(res_gdf, gpd.GeoDataFrame) and len(res_gdf):
    # Optionen
    copt1, copt2, copt3 = st.columns(3)
    with copt1:
        show_flurstks = st.checkbox("Flurstücke zeigen", value=True)
    with copt2:
        show_buildings = st.checkbox("Gebäude zeigen", value=False)
    with copt3:
        compute_lmax_all = st.checkbox(
            "Zulässigen Quellpegel (Lmax) je Standort berechnen",
            value=True,
            help="Lmax = maximaler Quellpegel, der an allen Nachbarn (inkl. Sicherheitsaufschlag) noch zulässig wäre."
        )

    # Helper: Lmax am Punkt (unabhängig von tatsächlicher Quelle)
    def max_source_db_at_point(
        p: Point,
        self_building_id,
        neighbors_tree,
        neighbor_geoms,
        neighbor_ids,
        allowed_db_list,
        geom_wkb_to_ix,   
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
            bound = (float(allowed_db_list[ix]) - safety_db) + (K_LN * np.log(sep) + C0)
            bounds.append(bound)
        return float(min(bounds)) if bounds else float("+inf")

    # Spalten & Merge für Tooltip
    bld_id_col = st.session_state["bld_id_col"]
    parc_id_col = st.session_state["parc_id_col"]
    noise_col  = st.session_state["bld_noise_emitted_col"]
    allow_col  = st.session_state["bld_allowed_at_facade_col"]

    bld_attrs = buildings_gdf[[bld_id_col, noise_col, allow_col]].copy()
    bld_attrs.columns = [bld_id_col, "src_L_db", "allowed_facade_db"]
    results = res_gdf.merge(bld_attrs, left_on="building_id", right_on=bld_id_col, how="left")

    # Nachbarstrukturen global
    neighbor_geoms = list(buildings_gdf.geometry.values)
    neighbor_ids   = list(buildings_gdf[bld_id_col].values)
    neighbors_tree = STRtree(neighbor_geoms)
    allowed_db_list = list(buildings_gdf[allow_col].astype(float).values)
    geom_wkb_to_ix = {g.wkb: i for i, g in enumerate(neighbor_geoms)}

    # Map: Gebäude-ID -> Geometrie (für eigene Wand im Tooltip)
    bld_geom_map = dict(zip(buildings_gdf[bld_id_col].values, buildings_gdf.geometry.values))

    # CRS-Flag (wie oben)
    try:
        crs_is_metric_map = not (buildings_gdf.crs.is_geographic if buildings_gdf.crs else True)
    except Exception:
        crs_is_metric_map = True


    # Lmax/Headroom optional berechnen (einmalig & merken)
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
                    # tatsächlicher L1_eff am Punkt inkl. dynamischem Wandaufschlag
                    if pd.notna(r.get("src_L_db")):
                        own_geom_here = bld_geom_map.get(r["building_id"])
                        L1_eff_pt, n_w_pt, gain_pt = effective_source_db_at_point(
                            float(r["src_L_db"]), pt, neighbors_tree, neighbor_geoms, geom_wkb_to_ix,
                            own_geom=own_geom_here, crs_is_metric=crs_is_metric_map
                        )
                        results.at[idx, "src_L_eff_db"] = L1_eff_pt
                        if np.isfinite(Lmax):
                            results.at[idx, "headroom_db"]  = Lmax - L1_eff_pt
                        results.at[idx, "nearby_walls"] = n_w_pt if "nearby_walls" not in results.columns else results.at[idx, "nearby_walls"]
                        results.at[idx, "wall_gain_db"] = gain_pt if "wall_gain_db" not in results.columns else results.at[idx, "wall_gain_db"]
        st.session_state["res_gdf"] = results
        st.session_state["lmax_computed"] = True
    else:
        results = st.session_state["res_gdf"]

    # Farben
    def status_color(s):
        s = (s or "").lower()
        if s == "irrelevant":
            return [0, 180, 0, 255]
        elif s == "ok":
            return [240, 200, 0, 255]
        elif s == "violation":
            return [220, 0, 0, 255]
        else:
            return [160, 160, 160, 200]

    results["color"] = results["status"].apply(status_color)

    # Punkte mit echter Geometrie
    pts = results[results.geometry.apply(lambda g: isinstance(g, Point))].copy()
    pts_wgs = None
    if len(pts) > 0:
        pts_wgs = pts.to_crs(4326).copy()
        pts_wgs["lat"] = pts_wgs.geometry.y
        pts_wgs["lon"] = pts_wgs.geometry.x

    # --- Ringe (Buffer) um die WP-Standorte ---
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
            if "building_id" in ring_df.columns: ring_df["building_id"] = ring_df["building_id"].astype(str)
            if "flurstk_id"   in ring_df.columns: ring_df["flurstk_id"]   = ring_df["flurstk_id"].astype(str)
            if "status"      in ring_df.columns: ring_df["status"]      = ring_df["status"].astype(str)
            if "required_clearance_m" in ring_df.columns:
                ring_df["required_clearance_m"] = pd.to_numeric(ring_df["required_clearance_m"], errors="coerce")

            rings_gdf = gpd.GeoDataFrame(ring_df, geometry=ring_geoms, crs=flurstks_gdf.crs)
            rings_gdf = rings_gdf[rings_gdf.geometry.notna() & (~rings_gdf.geometry.is_empty)]
            if len(rings_gdf) > 0:
                rings_wgs = rings_gdf.to_crs(4326)

    # ROTE PARZELLEN
    mask_no_solution = (results.geometry.isna() | results.geometry.apply(lambda g: not isinstance(g, Point))) & (results["status"] != "no-hp-due-to-threshold")
    no_sol_flurstk_ids = set(results.loc[mask_no_solution, "flurstk_id"].dropna().astype(str).values)
    red_flurstks = None
    if len(no_sol_flurstk_ids) > 0:
        red_flurstks = flurstks_gdf[flurstks_gdf[parc_id_col].astype(str).isin(no_sol_flurstk_ids)].copy()

    # WGS84-Layer vorbereiten
    layers = []

    # flurstks
    if show_flurstks:
        try:
            flurstks_wgs = flurstks_gdf.to_crs(4326)
            layers.append(pdk.Layer(
                "GeoJsonLayer",
                data=json.loads(flurstks_wgs.to_json()),
                stroked=True, filled=False,
                get_line_color=[50, 50, 50, 120], get_line_width=1
            ))
        except Exception:
            pass

    # Rote Parzellen
    if red_flurstks is not None and len(red_flurstks):
        red_wgs = red_flurstks.to_crs(4326)
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            data=json.loads(red_wgs.to_json()),
            stroked=True, filled=True,
            get_fill_color=[220, 0, 0, 60],
            get_line_color=[220, 0, 0, 200],
            get_line_width=2
        ))

    # Gebäude
    if show_buildings:
        try:
            bld_wgs = buildings_gdf.to_crs(4326)
            layers.append(pdk.Layer(
                "GeoJsonLayer",
                data=json.loads(bld_wgs.to_json()),
                stroked=False, filled=True,
                get_fill_color=[0, 120, 255, 40]
            ))
        except Exception:
            pass

    # Ringe
    if rings_wgs is not None and len(rings_wgs):
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            data=json.loads(rings_wgs.to_json()),
            stroked=True, filled=False,
            get_line_color=[128, 0, 200, 200],
            get_line_width=0.5
        ))

    # Ergebnis-Punkte
    if pts_wgs is not None and len(pts_wgs):
        # Tooltip inkl. Wandinfos
        tooltip_html = (
            "<b>Gebäude:</b> {building_id}<br/>"
            "<b>Status:</b> {status}<br/>"
            "<b>Reserve (dB):</b> {reserve_db}<br/>"
            "<b>Wände in ≤3 m (sichtbar):</b> {nearby_walls}<br/>"
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
            "lon","lat","color","building_id","flurstk_id","status","reserve_db",
            "nearby_walls","wall_gain_db",
            "src_L_db","src_L_eff_db",
            "dist_to_own_building_m","worst_neighbor_id","worst_neighbor_dist","required_clearance_m"
        ]
        if "Lmax_db" in pts_wgs.columns:      cols_for_map.append("Lmax_db")
        if "headroom_db" in pts_wgs.columns:  cols_for_map.append("headroom_db")

        df_plot = pts_wgs[cols_for_map].copy()

        # Typen säubern
        df_plot = df_plot.replace({np.inf: None, -np.inf: None})
        df_plot = df_plot.where(pd.notnull(df_plot), None)
        if "worst_neighbor_id" in df_plot.columns:
            df_plot["worst_neighbor_id"] = df_plot["worst_neighbor_id"].astype(str)

        def _to_builtin(x):
            if isinstance(x, list): return x
            if isinstance(x, np.generic): return x.item()
            return x

        for c in df_plot.columns:
            df_plot[c] = df_plot[c].map(_to_builtin)

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_plot,
            get_position=["lon","lat"],
            get_radius=2.0, radius_min_pixels=6, radius_max_pixels=10,
            get_fill_color="color",
            pickable=True
        ))

    # View
    if pts_wgs is not None and len(pts_wgs):
        center = union_all(pts_wgs.geometry.values).centroid
        vs_lat, vs_lon = center.y, center.x
    else:
        try:
            bbox = flurstks_gdf.to_crs(4326).total_bounds
            vs_lon = (bbox[0] + bbox[2]) / 2
            vs_lat = (bbox[1] + bbox[3]) / 2
        except Exception:
            vs_lat, vs_lon = 52.0, 9.0

    r = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vs_lat, longitude=vs_lon, zoom=15, bearing=0, pitch=0),
        map_style=None,
        tooltip={"html": tooltip_html, "style":{"backgroundColor":"white","color":"black"}}
    )
    st.pydeck_chart(r)

    st.caption(
    "Punkte: "
    "🟢 irrelevant (Reserve ≥ Schwelle) · "
    "🟡 ok (zulässig, aber < Irrelevanz-Schwelle oder Irrelevanz aus) · "
    "🔴 violation (beste gefundene Position, aber unzulässig) · "
    "⚪︎ grau = sonstiger Status.\n"
    "Flächen: 🔴 rot gefüllte Flurstücke = aktuell kein zulässiger Standort gefunden "
    "(Fälle mit Leistungs-Ausschluss sind davon ausgenommen). "
    "Ringe zeigen den erforderlichen Mindestabstand. "
    "Tooltips: Reserve, Wandanzahl/Aufschlag, Abstände, ggf. Lmax/Headroom."
)
else:
    st.info("Noch keine Ergebnisse zum Anzeigen. Bitte zuerst die Standortsuche ausführen.")



# ===================== 8) Export als GeoPackage ========================

st.divider()
st.header("8) Export")

res_gdf = st.session_state.get("res_gdf")
if isinstance(res_gdf, gpd.GeoDataFrame) and len(res_gdf):

    cexp1, cexp2 = st.columns([2,1])
    with cexp1:
        gpkg_name = st.text_input("Dateiname (ohne Endung)", value="wp_ergebnis",
                                  help="Die Endung .gpkg wird automatisch ergänzt.")
    with cexp2:
        export_btn = st.button("GeoPackage exportieren (Punkte, Ringe, rote Flurstücke)", use_container_width=True)

    if export_btn:
        try:
            # --- Spaltennamen aus der UI ---
            bld_id_col = st.session_state["bld_id_col"]
            parc_id_col = st.session_state["parc_id_col"]
            noise_col  = st.session_state["bld_noise_emitted_col"]
            allow_col  = st.session_state["bld_allowed_at_facade_col"]
            power_col  = st.session_state.get("bld_hp_power_col")
            use_power  = st.session_state.get("use_power_threshold", False) and power_col and power_col != "— keine Auswahl —" and power_col in buildings_gdf.columns

            # --- Attribute der Gebäude dazumergen (für Tooltips/Export) ---
            bld_attrs = [bld_id_col, noise_col, allow_col]
            if use_power:
                bld_attrs.append(power_col)
            bld_attr_df = buildings_gdf[bld_attrs].copy()
            rename_cols = [bld_id_col, "src_L_db", "allowed_facade_db"] + (["hp_power_kw"] if use_power else [])
            bld_attr_df.columns = rename_cols

            results = res_gdf.merge(bld_attr_df, left_on="building_id", right_on=bld_id_col, how="left")

            # --- Parameter-Spalten anreichern ---
            results["param_setback_m"] = st.session_state.get("setback_m", 0.0) if st.session_state.get("use_setback", False) else 0.0
            results["param_r_ref_m"]   = float(st.session_state.get("ref_distance_m", 0.0)) if "ref_distance_m" in st.session_state else float(0.0)
            results["param_min_r_m"]   = float(st.session_state.get("min_distance_m", 1.0)) if "min_distance_m" in st.session_state else float(1.0)
            results["param_safety_db"] = float(st.session_state.get("safety_margin_db", 0.0)) if "safety_margin_db" in st.session_state else float(0.0)
            results["crs_epsg"]        = flurstks_gdf.crs.to_epsg() if flurstks_gdf.crs else None

            # --- Geometrie-Quelle + Centroid-Fallback ---
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

            # Aufstell-/Hauben-Parameter (Info)
            pts_gdf["placement_choice"]   = str(st.session_state.get("placement_choice"))
            pts_gdf["placement_gain_db"]  = float(PLACEMENT_GAINS.get(st.session_state.get("placement_choice"), 0.0))
            pts_gdf["enclosure_atten_db"] = float(st.session_state.get("enclosure_atten_db", 0.0) or 0.0)

            # Irrelevanz-Infos
            irr_on = bool(st.session_state.get("use_irrelevance", False))
            irr_thr = float(st.session_state.get("irrelevance_threshold_db", 6.0)) if irr_on else np.nan
            if "irrelevant" not in pts_gdf.columns:
                pts_gdf["irrelevant"] = False
            pts_gdf["irrelevant"] = pts_gdf["irrelevant"].astype(bool).astype(int)
            pts_gdf["irr_threshold_db"] = irr_thr

            # ==== Auswahl & Typ-Glättung ====
            keep = [c for c in [
                "building_id","flurstk_id","status","reserve_db",
                "nearby_walls","wall_gain_db",
                "worst_neighbor_id","worst_neighbor_dist",
                "dist_to_own_building_m","required_clearance_m",
                "src_L_db","src_L_eff_db","allowed_facade_db","hp_power_kw",
                "placement_choice","placement_gain_db","enclosure_atten_db",
                "irrelevant","irr_threshold_db",
                "param_setback_m","param_r_ref_m","param_min_r_m","param_safety_db",
                "crs_epsg","geom_source",
                "Lmax_db","headroom_db"
            ] if c in pts_gdf.columns]

            pts_out = pts_gdf[keep + ["geometry"]].copy()

            # Strings
            for c in ["building_id","flurstk_id","worst_neighbor_id","status","geom_source","placement_choice"]:
                if c in pts_out.columns:
                    pts_out[c] = pts_out[c].astype("string")

            # Numerik
            for c in ["reserve_db","worst_neighbor_dist","dist_to_own_building_m","required_clearance_m",
                      "src_L_db","src_L_eff_db","allowed_facade_db","hp_power_kw",
                      "placement_gain_db","enclosure_atten_db",
                      "irr_threshold_db","nearby_walls","wall_gain_db",
                      "param_setback_m","param_r_ref_m","param_min_r_m","param_safety_db",
                      "crs_epsg","Lmax_db","headroom_db"]:
                if c in pts_out.columns:
                    pts_out[c] = pd.to_numeric(pts_out[c], errors="coerce")

            if "irrelevant" in pts_out.columns:
                pts_out["irrelevant"] = pts_out["irrelevant"].astype("Int64")

            # --- Schreiben: Punkte-Layer ---
            outdir = Path(st.session_state.get("_tmpdir", "uploaded_data"))
            outdir.mkdir(parents=True, exist_ok=True)

            fname = (gpkg_name or "wp_ergebnis").strip()
            if not fname:
                fname = "wp_ergebnis"
            gpkg_path = outdir / f"{fname}.gpkg"

            if gpkg_path.exists():
                try:
                    gpkg_path.unlink()
                except Exception:
                    gpkg_path = outdir / f"{fname}_{np.random.randint(1e6)}.gpkg"

            pts_out.to_file(gpkg_path, driver="GPKG", layer="wp_results")

            # --- Ringe-Layer ---
            if "required_clearance_m" in pts_gdf.columns:
                ring_rows = pts_gdf[
                    pts_gdf.geometry.notna() &
                    (~pts_gdf.geometry.is_empty) &
                    pd.to_numeric(pts_gdf["required_clearance_m"], errors="coerce").notna()
                ].copy()

                if len(ring_rows):
                    ring_geoms = []
                    for _, rr in ring_rows.iterrows():
                        try:
                            ring_geoms.append(rr.geometry.buffer(float(rr["required_clearance_m"]), resolution=32))
                        except Exception:
                            ring_geoms.append(None)
                    rings_gdf = gpd.GeoDataFrame(
                        ring_rows[["building_id","flurstk_id","status","required_clearance_m","irrelevant"]]
                            .astype({"building_id":"string","flurstk_id":"string","status":"string"}),
                        geometry=ring_geoms, crs=flurstks_gdf.crs
                    )
                    if "irrelevant" in rings_gdf.columns:
                        rings_gdf["irrelevant"] = rings_gdf["irrelevant"].astype(bool).astype(int)
                    rings_gdf = rings_gdf[rings_gdf.geometry.notna() & (~rings_gdf.geometry.is_empty)]
                    if len(rings_gdf):
                        rings_gdf["required_clearance_m"] = pd.to_numeric(rings_gdf["required_clearance_m"], errors="coerce")
                        rings_gdf.to_file(gpkg_path, driver="GPKG", layer="wp_rings")

            # --- Rote Flurstücke-Layer ---
            mask_no_solution = (res_gdf.geometry.isna() |
                                res_gdf.geometry.apply(lambda g: not isinstance(g, Point))) & \
                               (res_gdf["status"] != "no-hp-due-to-threshold")
            no_sol_ids = set(res_gdf.loc[mask_no_solution, "flurstk_id"].dropna().astype(str).values)

            if len(no_sol_ids) > 0:
                red_flurstks_gdf = flurstks_gdf[flurstks_gdf[parc_id_col].astype(str).isin(no_sol_ids)].copy()
                red_flurstks_gdf["no_solution"] = 1
                red_flurstks_gdf[parc_id_col] = red_flurstks_gdf[parc_id_col].astype("string")
                red_flurstks_gdf.to_file(gpkg_path, driver="GPKG", layer="wp_flurstks_nosolution")

            # --- Download ---
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

else:
    st.info("Keine Ergebnisse zum Export vorhanden. Bitte zuerst die Standortsuche ausführen.")
