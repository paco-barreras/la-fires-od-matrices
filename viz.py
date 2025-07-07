import pydeck as pdk
import numpy as np
import pandas as pd

def build_od_map_flat(
    polygons, od_df, *,
    origin="zipcode_origin", dest="zipcode_dest",
    weight="count", id_col="geography_id",
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"  # <-- nicer default
):
    """
    2-D choropleth of destination ZIPs.
    Hover shows weight from every origin present in od_df.
    """

    # polygons actually used
    dest_ids = od_df[dest].unique()
    poly = polygons[polygons[id_col].isin(dest_ids)].copy()

    # aggregate by destination
    dest_tot = od_df.groupby(dest)[weight].sum().rename("w_dest")

    # colour ramp (light→dark blue)
    def ramp(t):
        lo, hi = np.array([199, 233, 255]), np.array([0, 76, 153])
        return (lo + (hi - lo) * t).astype(int).tolist()

    mx = dest_tot.max() or 1
    poly["w_dest"] = poly[id_col].map(dest_tot).fillna(0)
    poly["fill"] = poly["w_dest"].apply(lambda v: ramp(v / mx))

    # pivot so tooltip can show weights per origin
    pivot = (
        od_df.pivot_table(index=dest, columns=origin, values=weight, aggfunc="sum")
             .fillna(0)
    )
    origin_cols = pivot.columns.tolist()
    poly = poly.join(pivot, on=id_col)

    # smarter formatting
    def fmt(val):
        if isinstance(val, (int, np.integer)) or np.isclose(val % 1, 0):
            return f"{int(val)}"
        return f"{val:.1f}%"

    def tooltip_text(row):
        parts = [f"{c}: {fmt(row[c])}" for c in origin_cols if row[c] > 0]
        return "<br/>".join(parts) or "0"

    poly["tooltip"] = poly.apply(tooltip_text, axis=1)

    layer = pdk.Layer(
        "GeoJsonLayer",
        poly,
        get_fill_color="fill",
        get_line_color=[60, 60, 60],
        opacity=0.65,
        pickable=True,
        auto_highlight=True,
    )

    view = pdk.ViewState(
        latitude=poly.geometry.centroid.y.mean(),
        longitude=poly.geometry.centroid.x.mean(),
        zoom=8,
        pitch=0,
        bearing=0,
    )

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"html": "{tooltip}", "style": {"color": "white"}},
        map_style=map_style,
    )

def build_od_map_pdk(
    polygons, moved, *,
    origin="zipcode_origin", dest="zipcode_dest",
    pct="percentage", id_col="geography_id",
    min_flow=0.05
):
    """PyDeck map: polygons + arcs whose colour & width follow `pct`."""

    def ramp(t):
        """Blue colour ramp, t ∈ [0,1] → RGB list."""
        lo = np.array([199, 233, 255], dtype=int)   # light
        hi = np.array([0, 76, 153], dtype=int)      # dark
        return (lo + (hi - lo) * t).astype(int).tolist()

    # keep only ZIPs present in OD table
    ids = pd.unique(pd.concat([moved[origin], moved[dest]]))
    poly = polygons[polygons[id_col].isin(ids)].copy()

    # centroids
    poly["lon"] = poly.geometry.centroid.x
    poly["lat"] = poly.geometry.centroid.y
    coord = poly.set_index(id_col)[["lon", "lat"]]

    # flows with geometry
    flows = (
        moved.merge(coord, left_on=origin, right_index=True)
             .rename(columns={"lon": "o_lon", "lat": "o_lat"})
             .merge(coord, left_on=dest,   right_index=True)
             .rename(columns={"lon": "d_lon", "lat": "d_lat"})
    )
    flows = flows[flows[pct] >= min_flow].copy()
    flows["source"] = flows[["o_lon", "o_lat"]].values.tolist()
    flows["target"] = flows[["d_lon", "d_lat"]].values.tolist()

    # width & colour from percentage
    max_p = flows[pct].max()
    flows["width"] = (flows[pct] / max_p * 25).clip(1)
    flows["color"] = flows[pct].apply(lambda v: ramp(v / max_p))

    arc_layer = pdk.Layer(
        "ArcLayer",
        flows.assign(p=flows[pct].round(2)),
        get_source_position="source",
        get_target_position="target",
        get_width="width",
        get_source_color="color",
        get_target_color="color",
        pickable=True,
        auto_highlight=True,
    )

    # polygon colour by destination marginal (same ramp)
    dest_marg = moved.groupby(dest)[pct].sum()
    poly["density"] = poly[id_col].map(dest_marg).fillna(0)
    max_d = poly["density"].max() or 1
    poly["fill"] = poly["density"].apply(lambda v: ramp(v / max_d))

    poly_layer = pdk.Layer(
        "GeoJsonLayer",
        poly,
        get_fill_color="fill",
        get_line_color=[60, 60, 60],
        opacity=0.4,
    )

    view = pdk.ViewState(
        latitude=flows["o_lat"].mean(),
        longitude=flows["o_lon"].mean(),
        zoom=8,
        pitch=35,
    )
    tooltip = {"html": f"{{{origin}}} → {{{dest}}}<br/>{{p}} %", "style": {"color": "white"}}

    return pdk.Deck(layers=[poly_layer, arc_layer], initial_view_state=view, tooltip=tooltip)