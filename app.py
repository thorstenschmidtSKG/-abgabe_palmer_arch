import numpy as np
import pandas as pd
import streamlit as st

from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    CustomJS,
    NumeralTickFormatter,
    Label,
    LabelSet,
    Span,          # ✅ NEW
)
from bokeh.plotting import figure



# Config
DATA_PATH = "penguins_size.csv"

# Set Color Palette
PALETTE = [
    "#4E79A7",  # muted blue (primary)
    "#59A14F",  # muted green
    "#9C755F",  # muted brown
    "#F28E2B",  # muted orange (use sparingly)
    "#B07AA1",  # muted purple
    "#76B7B2",  # muted teal
]

# Define visual hierarchy
BG = "#FFFFFF"
TEXT = "#1F2937"
GRID = "#E5E7EB"

CONTEXT_FILL = "#D1D5DB"
CONTEXT_ALPHA = 0.25
FOCUS_ALPHA = 0.95

# Fixed sizes (due to usage of multiple graphs)
W_MAIN = 760
H_MAIN = 460
W_RIGHT = 240
H_TOP = 150

st.set_page_config(page_title="Penguins • Story-first Bokeh", layout="wide")



# Data utilities
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # remap to match inital dataset
    rename_map = {
        "culmen_length_mm": "bill_length_mm",
        "culmen_depth_mm": "bill_depth_mm",
        "body_mass_(g)": "body_mass_g",
        "flipper_length_(mm)": "flipper_length_mm",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # clean missing values
    for c in ["species", "island", "sex"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].str.lower().isin(["nan", "none", "null", ""]), c] = np.nan
    return df


def compute_hist(values: np.ndarray, vmin: float, vmax: float, nbins: int):
    edges = np.linspace(vmin, vmax, nbins + 1)
    counts, _ = np.histogram(values, bins=edges)
    left = edges[:-1]
    right = edges[1:]
    return left, right, counts


def style_plot(p: figure, *, show_xgrid=True, show_ygrid=True):
    p.background_fill_color = BG
    p.border_fill_color = BG
    p.outline_line_color = None

    p.xgrid.visible = show_xgrid
    p.ygrid.visible = show_ygrid
    p.xgrid.grid_line_color = GRID
    p.ygrid.grid_line_color = GRID
    p.xgrid.grid_line_alpha = 0.35 if show_xgrid else 0.0
    p.ygrid.grid_line_alpha = 0.35 if show_ygrid else 0.0
    p.xgrid.grid_line_width = 1
    p.ygrid.grid_line_width = 1

    for ax in p.axis:
        ax.axis_line_color = None
        ax.major_tick_line_color = None
        ax.minor_tick_line_color = None
        ax.major_label_text_color = TEXT
        ax.axis_label_text_color = TEXT
        ax.major_label_text_font_size = "10pt"
        ax.axis_label_text_font_size = "10pt"

    p.title.text_color = TEXT
    p.title.text_font_size = "12pt"
    p.title.text_font_style = "bold"



# Load + validate for matching headers
df = load_data(DATA_PATH)

required = ["bill_length_mm", "bill_depth_mm", "species", "island", "sex"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

df["bill_length_mm"] = pd.to_numeric(df["bill_length_mm"], errors="coerce")
df["bill_depth_mm"] = pd.to_numeric(df["bill_depth_mm"], errors="coerce")



# Sidebar:
with st.sidebar:
    st.header("Filters")
    drop_na = st.checkbox("Drop rows with missing values", value=True)

    dff = df.copy()
    if drop_na:
        dff = dff.dropna(subset=["bill_length_mm", "bill_depth_mm", "species", "island", "sex"])

    species_all = sorted(dff["species"].dropna().unique().tolist())
    island_all = sorted(dff["island"].dropna().unique().tolist())
    sex_all = sorted(dff["sex"].dropna().unique().tolist())

    species_sel = st.multiselect("Species", species_all, default=species_all)
    island_sel = st.multiselect("Island", island_all, default=island_all)
    sex_sel = st.multiselect("Sex", sex_all, default=sex_all)

    st.divider()
    st.header("Story focus")

    focus_species = st.selectbox(
        "Highlight species",
        options=species_all,
        index=(species_all.index("Gentoo") if "Gentoo" in species_all else 0),
    )

    focus_color = st.selectbox(
        "Highlight color",
        options=PALETTE,
        index=0,
    )

    nbins = st.slider("Histogram bins", 10, 60, 25, 1)



# Apply filters
dff = dff[
    dff["species"].isin(species_sel)
    & dff["island"].isin(island_sel)
    & dff["sex"].isin(sex_sel)
].copy()

dff = dff.dropna(subset=["bill_length_mm", "bill_depth_mm", "species", "island", "sex"])
if dff.empty:
    st.warning("No data after filters.")
    st.stop()

# Focus columns
dff["is_focus"] = dff["species"].eq(focus_species)
dff["focus_alpha"] = np.where(dff["is_focus"], FOCUS_ALPHA, 0.0)

# Add context to chart
st.markdown(
    f"""
# Penguins — physical traits differentiate species

**Context:**  
A penguin’s *bill* is its beak. Here we compare **bill length** (how long) and **bill depth** (how thick).  

**Big idea:**  
*{focus_species}* penguins occupy a **distinct region** in this space, meaning their beak shape clearly differs
from other species.

**How to explore:**  
Use **box or lasso selection** in the scatter plot to highlight subsets and see how the distributions change.
"""
)



# histogram markers for focus-species
focus_df = dff[dff["species"] == focus_species]
focus_len = float(focus_df["bill_length_mm"].median()) if not focus_df.empty else float(dff["bill_length_mm"].median())
focus_dep = float(focus_df["bill_depth_mm"].median()) if not focus_df.empty else float(dff["bill_depth_mm"].median())



# Sources + ranges
x = dff["bill_length_mm"].to_numpy()
y = dff["bill_depth_mm"].to_numpy()

x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

x_pad = (x_max - x_min) * 0.06 if x_max > x_min else 1.0
y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 1.0

x_range = (x_min - x_pad, x_max + x_pad)
y_range = (y_min - y_pad, y_max + y_pad)

source = ColumnDataSource(dff)



# Histograms (all + selected overlay)
x_left, x_right, x_all = compute_hist(x, x_min, x_max, nbins)
y_left, y_right, y_all = compute_hist(y, y_min, y_max, nbins)

hist_x = ColumnDataSource(dict(left=x_left, right=x_right, all=x_all, sel=np.zeros_like(x_all)))
hist_y = ColumnDataSource(dict(left=y_left, right=y_right, all=y_all, sel=np.zeros_like(y_all)))



# Direct label positions
label_df = (
    dff.groupby("species", dropna=True)[["bill_length_mm", "bill_depth_mm"]]
    .median()
    .reset_index()
)
label_source = ColumnDataSource(label_df)



# Plots (fixed sizes)
TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"

p_scatter = figure(
    width=W_MAIN,
    height=H_MAIN,
    tools=TOOLS,
    toolbar_location="right",
    x_range=(*x_range,),
    y_range=(*y_range,),
    title="Bill length vs bill depth",
)
style_plot(p_scatter, show_xgrid=True, show_ygrid=True)

p_scatter.xaxis.axis_label = "Bill length (mm)"
p_scatter.yaxis.axis_label = "Bill depth (mm)"
p_scatter.xaxis.formatter = NumeralTickFormatter(format="0")
p_scatter.yaxis.formatter = NumeralTickFormatter(format="0")

# Context layer: all points (muted)
p_scatter.circle(
    x="bill_length_mm",
    y="bill_depth_mm",
    source=source,
    size=6,
    fill_color=CONTEXT_FILL,
    fill_alpha=CONTEXT_ALPHA,
    line_color=None,
    nonselection_fill_alpha=CONTEXT_ALPHA,
)

# Focus layer to only highlight one species
focus_renderer = p_scatter.circle(
    x="bill_length_mm",
    y="bill_depth_mm",
    source=source,
    size=7,
    fill_color=focus_color,
    fill_alpha="focus_alpha",
    line_color=None,
)

hover = HoverTool(
    tooltips=[
        ("species", "@species"),
        ("island", "@island"),
        ("sex", "@sex"),
        ("bill_length", "@bill_length_mm{0.0} mm"),
        ("bill_depth", "@bill_depth_mm{0.0} mm"),
    ],
    renderers=[focus_renderer],
)
p_scatter.add_tools(hover)

labels = LabelSet(
    x="bill_length_mm",
    y="bill_depth_mm",
    text="species",
    source=label_source,
    text_color=TEXT,
    text_font_size="10pt",
    text_font_style="bold",
    x_offset=6,
    y_offset=6,
)
p_scatter.add_layout(labels)

ann = Label(
    x=x_range[0] + 0.05 * (x_range[1] - x_range[0]),
    y=y_range[1] - 0.08 * (y_range[1] - y_range[0]),
    x_units="data",
    y_units="data",
    text=f"Highlight: {focus_species}",
    text_color=TEXT,
    text_font_size="10pt",
    background_fill_color=BG,
    background_fill_alpha=0.9,
    border_line_color=None,
)
p_scatter.add_layout(ann)



# Top histogram
p_hist_x = figure(
    width=W_MAIN,
    height=H_TOP,
    tools="",
    toolbar_location=None,
    x_range=p_scatter.x_range,
    title="Bill length distribution (all vs selected)",
)
style_plot(p_hist_x, show_xgrid=False, show_ygrid=True)
p_hist_x.yaxis.axis_label = "Count"
p_hist_x.xaxis.visible = False

p_hist_x.quad(
    top="all", left="left", right="right", bottom=0, source=hist_x,
    fill_color=CONTEXT_FILL, fill_alpha=0.55, line_color=None
)
p_hist_x.quad(
    top="sel", left="left", right="right", bottom=0, source=hist_x,
    fill_color=focus_color, fill_alpha=0.85, line_color=None
)

# focus species marker on bill-length histogram
p_hist_x.add_layout(Span(
    location=focus_len, dimension="height",
    line_color=focus_color, line_width=2, line_dash="dashed"
))

# Right histogram (bill depth)
p_hist_y = figure(
    width=W_RIGHT,
    height=H_MAIN,
    tools="",
    toolbar_location=None,
    y_range=p_scatter.y_range,
    title="Bill depth distribution",
)
style_plot(p_hist_y, show_xgrid=True, show_ygrid=False)
p_hist_y.xaxis.axis_label = "Count"
p_hist_y.yaxis.visible = False

p_hist_y.quad(
    left=0, right="all", bottom="left", top="right", source=hist_y,
    fill_color=CONTEXT_FILL, fill_alpha=0.55, line_color=None
)
p_hist_y.quad(
    left=0, right="sel", bottom="left", top="right", source=hist_y,
    fill_color=focus_color, fill_alpha=0.85, line_color=None
)

# focus species marker on bill-depth histogram
p_hist_y.add_layout(Span(
    location=focus_dep, dimension="width",
    line_color=focus_color, line_width=2, line_dash="dashed"
))



# Linked selection -> update selected hist overlays
callback = CustomJS(
    args=dict(source=source, hist_x=hist_x, hist_y=hist_y),
    code="""
    function histCounts(values, edges) {
        const counts = new Array(edges.length - 1).fill(0);
        for (let i = 0; i < values.length; i++) {
            const v = values[i];
            if (v == null || isNaN(v)) continue;
            for (let b = 0; b < edges.length - 1; b++) {
                if (v >= edges[b] && (v < edges[b+1] || (b === edges.length-2 && v <= edges[b+1]))) {
                    counts[b] += 1;
                    break;
                }
            }
        }
        return counts;
    }

    const inds = source.selected.indices;
    const x = source.data["bill_length_mm"];
    const y = source.data["bill_depth_mm"];

    const x_left = hist_x.data["left"];
    const x_right = hist_x.data["right"];
    const y_left = hist_y.data["left"];
    const y_right = hist_y.data["right"];

    const x_edges = [x_left[0]];
    for (let i = 0; i < x_right.length; i++) x_edges.push(x_right[i]);

    const y_edges = [y_left[0]];
    for (let i = 0; i < y_right.length; i++) y_edges.push(y_right[i]);

    const xs = [];
    const ys = [];
    for (let i = 0; i < inds.length; i++) {
        const idx = inds[i];
        xs.push(x[idx]);
        ys.push(y[idx]);
    }

    hist_x.data["sel"] = histCounts(xs, x_edges);
    hist_y.data["sel"] = histCounts(ys, y_edges);

    hist_x.change.emit();
    hist_y.change.emit();
"""
)
source.selected.js_on_change("indices", callback)



# Layout + render
layout = gridplot(
    [
        [None, p_hist_x],
        [p_hist_y, p_scatter],
    ],
    merge_tools=False,
)

left, center, right = st.columns([1, 3, 1])
with center:
    st.bokeh_chart(layout, use_container_width=False)

with st.expander("Show filtered data (optional)"):
    st.dataframe(dff.reset_index(drop=True), use_container_width=True)