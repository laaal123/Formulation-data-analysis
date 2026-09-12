"""Prompt 7 -- Streamlit interface.

Tab order is the order the analysis must be done in, and the tabs are gated. Run:
    streamlit run app.py
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from catalogue import BY_ID, GROUPS, available_analyses, small_batch_guidance
from config import FAMILIES, RunSettings
from errors import DataError, Refusal
from explore import (
    compare_batches,
    compare_profiles,
    descriptive_summary,
    profile_summary,
    spec_check,
    trend_table,
    what_changed,
)
from gating import (
    ATTRIBUTION_IDS,
    FINDINGS_BANNER,
    QUICK_IDS,
    TABS,
    SessionState,
    state_from_result,
    tab_status,
)
from interpret import linear_contributions, partial_dependence
from loader import (
    blank_entry_table,
    build_dataset,
    long_to_wide,
    read_table,
    split_single_table,
)
from modelling import MODEL_TYPES
from pipeline import run_attribution
from record import to_json, to_markdown
from response import PROFILE_METRICS
from synthetic import example_dataset
from variables import (
    DUAL_ROLE,
    FAMILY_MEANING,
    build_template,
    count_variables,
    dissolution_template,
    families_for,
    profile_frame,
    reference_frame,
)
from validation import CV_SCHEMES

st.set_page_config(page_title="Batch factor attribution", layout="wide")

S = st.session_state
S.setdefault("ds", None)
S.setdefault("result", None)
S.setdefault("diagnostics_viewed", False)
S.setdefault("response_choice", None)
S.setdefault("model_configured", False)
S.setdefault("analysis_id", "")
S.setdefault("entry_df", None)


# ---------------------------------------------------------------------
def _state() -> SessionState:
    if S.result is not None:
        st_ = state_from_result(S.result, S.diagnostics_viewed)
        st_.analysis_id = S.get("analysis_id", "attribution")
        st_.analysis_chosen = bool(st_.analysis_id)
        st_.quick_ready = st_.analysis_id in QUICK_IDS
        return st_
    return SessionState(
        data_loaded=S.ds is not None,
        data_has_errors=bool(S.ds is not None and not S.ds.report.ok),
        diagnostics_viewed=S.diagnostics_viewed,
        response_chosen=S.response_choice is not None,
        response_usable=S.response_choice is not None,
        model_configured=S.model_configured,
        analysis_chosen=bool(S.get("analysis_id")),
        analysis_id=S.get("analysis_id", ""),
        quick_ready=S.get("analysis_id", "") in QUICK_IDS,
    )


def _heatmap(corr: pd.DataFrame, order: list[str], title: str):
    sub = corr.reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(min(12, 0.22 * len(order) + 3),) * 2)
    im = ax.imshow(sub.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
st.title("Batch factor attribution")
st.caption(
    "Development and investigation aid. Not a validated GMP output. Findings are "
    "associations with their confounders named, plus the experiment that would settle them."
)

status = tab_status(_state())
labels = {t: (t if status[t]["unlocked"] else f"{t}  (locked)") for t in TABS}
choice = st.sidebar.radio("Steps", list(TABS), format_func=lambda t: labels[t])
st.sidebar.markdown("---")
if S.ds is not None:
    st.sidebar.write(f"{S.ds.n_batches} batches, {S.ds.n_formulations} formulations")
    st.sidebar.write(f"{len(S.ds.predictor_columns())} usable predictors")

if not status[choice]["unlocked"]:
    st.subheader(choice)
    st.warning(f"Locked. {status[choice]['reason']}")
    st.stop()


# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
if choice == "Data":
    st.subheader("1. Data")

    with st.expander("What can I analyse? Every variable this app understands", expanded=False):
        counts = count_variables()
        st.write(
            f"{sum(counts.values())} named variables across five families, plus "
            f"{len(profile_frame())} metrics computed from a dissolution profile. Any "
            "numeric column you add will be analysed even if it is not on this list; what "
            "the family tag changes is how it is treated."
        )
        for fam, meaning in FAMILY_MEANING.items():
            st.markdown(f"**{fam}** ({counts.get(fam, 0)} listed) — {meaning}")
        st.markdown("---")
        pick_fams = st.multiselect(
            "Show families", list(FAMILY_MEANING), default=list(FAMILY_MEANING)
        )
        ref = reference_frame(pick_fams)
        st.dataframe(ref, use_container_width=True, hide_index=True, height=340)
        st.caption(
            "Listed in two families on purpose: "
            + ", ".join(DUAL_ROLE)
            + ". They are results you measured, and they can also be the thing you are "
            "trying to explain. They arrive tagged IPQC_PHYSICAL; retag as RESPONSE if that "
            "is your question."
        )
        st.markdown("**Computed from a dissolution profile table**")
        st.dataframe(profile_frame(), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Build a blank template from the variables you actually record**")
        chosen = st.multiselect(
            "Pick your variables", ref["variable"].tolist(),
            default=[v for v in ["polymer_level_pct", "granulation_water_pct",
                                 "main_compression_kN", "hardness_N", "q30_pct"]
                     if v in ref["variable"].tolist()],
        )
        n_t = st.number_input("Rows (batches)", 1, 200, 12, key="tmpl_rows")
        if chosen:
            tmpl = build_template(chosen, int(n_t))
            st.dataframe(tmpl.head(4), use_container_width=True, hide_index=True)
            st.download_button(
                "Download batch table template (CSV)",
                tmpl.to_csv(index=False), "batch_table_template.csv",
            )
            st.download_button(
                "Download dissolution template (CSV)",
                dissolution_template(tmpl["batch"].tolist()).to_csv(index=False),
                "dissolution_template.csv",
            )
            st.caption(
                "Families that will be applied: "
                + ", ".join(f"{k} = {v}" for k, v in families_for(chosen).items())
            )

    mode = st.radio(
        "How do you want to get your data in?",
        ["Load the example", "Upload files", "Type it in", "Paste from Excel"],
        horizontal=True,
    )

    # ---------------- example -------------------------------------
    if mode == "Load the example":
        st.caption(
            "24 batches across 4 formulations, with dissolution profiles and confounding "
            "built in on purpose. Useful for seeing what each screen does."
        )
        if st.button("Load the synthetic example"):
            S.ds = example_dataset()
            S.result = None
            S.diagnostics_viewed = False
            S.analysis_id = ""
            st.success("Loaded.")

    # ---------------- upload --------------------------------------
    elif mode == "Upload files":
        st.caption(
            "One row per batch. If your batch table already has a `formulation` column, "
            "the metadata file is optional."
        )
        up_wide = st.file_uploader("Batch table (CSV or Excel)", type=["csv", "xlsx"])
        up_meta = st.file_uploader("Batch metadata (optional if formulation is above)",
                                   type=["csv", "xlsx"])
        up_diss = st.file_uploader(
            "Dissolution profiles (optional): batch, medium, time_h, percent_released",
            type=["csv", "xlsx"],
        )
        fmt = st.radio("Batch table format", ["wide", "long"], horizontal=True)
        if up_wide is not None and st.button("Load and validate"):
            try:
                wide = read_table(up_wide)
                if fmt == "long":
                    wide = long_to_wide(wide)
                diss = read_table(up_diss) if up_diss is not None else None
                if up_meta is not None:
                    meta = read_table(up_meta)
                else:
                    wide, meta = split_single_table(wide)
                families = {c: "PROCESS" for c in wide.columns if c != "batch"}
                S.ds = build_dataset(wide, families, meta, diss)
                S.result = None
                S.diagnostics_viewed = False
                S.analysis_id = ""
                st.info("Loaded with every column tagged PROCESS. Correct the tags below.")
            except (Refusal, DataError, ValueError, KeyError) as exc:
                st.error(f"Refused to load: {exc}")

    # ---------------- manual entry --------------------------------
    elif mode == "Type it in":
        st.caption(
            "Build the table here. You need a `batch` column and a `formulation` column; "
            "add one column per thing you measured or set. Two batches is enough for the "
            "compare-and-describe analyses."
        )
        c1, c2 = st.columns([1, 2])
        n_rows = c1.number_input("Number of batches", 1, 200, 6)
        if "entry_cols" not in S:
            S.entry_cols = {"hardness_N": "IPQC_PHYSICAL", "assay_pct": "RESPONSE"}

        with c2.form("add_col", clear_on_submit=True):
            a, b, c = st.columns([2, 2, 1])
            new_name = a.text_input("Add a column", placeholder="e.g. hpmc_level_pct")
            new_fam = b.selectbox("Family", list(FAMILIES), index=list(FAMILIES).index("PROCESS"))
            if c.form_submit_button("Add") and new_name.strip():
                S.entry_cols[new_name.strip()] = new_fam
                S.entry_df = None

        drop = st.multiselect("Remove columns", list(S.entry_cols))
        if drop and st.button("Remove selected"):
            for d in drop:
                S.entry_cols.pop(d, None)
            S.entry_df = None

        base = blank_entry_table(int(n_rows), list(S.entry_cols))
        if S.get("entry_df") is not None:
            prev = S.entry_df
            for col in base.columns:
                if col in prev.columns:
                    base[col] = list(prev[col][: len(base)]) + [None] * max(0, len(base) - len(prev))
        S.entry_df = st.data_editor(
            base, num_rows="fixed", use_container_width=True, height=320, key="entry_editor"
        )
        st.caption(
            "Tip: you can paste a block straight from Excel into this grid. "
            "Column families: " + ", ".join(f"{k} = {v}" for k, v in S.entry_cols.items())
        )
        if st.button("Build dataset from this table", type="primary"):
            try:
                df = S.entry_df.copy()
                for col in S.entry_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                wide, meta = split_single_table(df)
                S.ds = build_dataset(wide, dict(S.entry_cols), meta)
                S.result = None
                S.diagnostics_viewed = False
                S.analysis_id = ""
                st.success(f"Built: {S.ds.n_batches} batches, {S.ds.n_formulations} formulations.")
            except (Refusal, DataError, ValueError, KeyError) as exc:
                st.error(f"Refused: {exc}")

    # ---------------- paste ---------------------------------------
    else:
        st.caption(
            "Copy the block out of Excel including the header row and paste it here. "
            "Tabs or commas both work. Must include `batch` and `formulation` columns."
        )
        text = st.text_area("Paste your table", height=240,
                            placeholder="batch\tformulation\thpmc_level_pct\tq30_pct\nB001\tF1\t18.2\t26.1")
        diss_text = st.text_area(
            "Dissolution profiles (optional): batch, medium, time_h, percent_released",
            height=120,
        )
        if st.button("Read pasted data") and text.strip():
            try:
                df = pd.read_csv(io.StringIO(text.strip()), sep=None, engine="python")
                diss = (
                    pd.read_csv(io.StringIO(diss_text.strip()), sep=None, engine="python")
                    if diss_text.strip() else None
                )
                wide, meta = split_single_table(df)
                families = {c: "PROCESS" for c in wide.columns if c != "batch"}
                S.ds = build_dataset(wide, families, meta, diss)
                S.result = None
                S.diagnostics_viewed = False
                S.analysis_id = ""
                st.info(
                    f"Read {S.ds.n_batches} batches. Every column is tagged PROCESS; "
                    "correct the tags below."
                )
            except (Refusal, DataError, ValueError, KeyError) as exc:
                st.error(f"Could not read that: {exc}")

    # ---------------- tagging and report --------------------------
    if S.ds is not None:
        st.markdown("---")
        st.markdown("**Column family tagging** — the family decides how a column is treated.")
        st.caption(
            "MATERIAL and FORMULATION and PROCESS are things you set or receive. "
            "IPQC_PHYSICAL is something you measured on the way through. RESPONSE is the "
            "result you want to explain."
        )
        tags = pd.DataFrame(
            {
                "column": list(S.ds.wide.columns),
                "family": [S.ds.families.get(c, "META") for c in S.ds.wide.columns],
                "excluded_reason": [S.ds.excluded.get(c, "") for c in S.ds.wide.columns],
            }
        )
        edited = st.data_editor(
            tags,
            column_config={
                "family": st.column_config.SelectboxColumn(options=list(FAMILIES)),
                "excluded_reason": st.column_config.TextColumn(disabled=True),
            },
            hide_index=True, use_container_width=True, height=300,
        )
        if st.button("Apply tags"):
            S.ds.families = dict(zip(edited["column"], edited["family"]))
            S.result = None
            st.success("Tags applied.")

        st.markdown("**Validation report**")
        st.dataframe(S.ds.report.to_frame(), use_container_width=True, hide_index=True)
        if not S.ds.report.ok:
            st.error("Errors above must be fixed in the source data. Nothing is coerced for you.")

        with st.expander("Missingness and explicit imputation"):
            st.dataframe(S.ds.missingness().head(20), use_container_width=True)
            method = st.selectbox("Method", ["median", "mean", "drop_columns", "drop_batches"])
            cols = st.multiselect(
                "Columns", [c for c in S.ds.wide.columns if S.ds.wide[c].isna().any()]
            )
            if st.button("Impute") and cols:
                S.ds.impute(method, cols)
                S.result = None
                st.success(f"{method} applied to {len(cols)} column(s) and recorded.")

        st.download_button(
            "Download this batch table as CSV",
            S.ds.wide.reset_index().to_csv(index=False),
            "batch_table.csv",
        )


# ---------------------------------------------------------------------
# 2. Choose analysis
# ---------------------------------------------------------------------
elif choice == "Choose analysis":
    st.subheader("2. Choose analysis")
    ds = S.ds
    st.info(small_batch_guidance(ds.n_batches, ds.n_formulations))

    rows = available_analyses(ds)
    for group in GROUPS:
        st.markdown(f"### {group}")
        for row in [r for r in rows if r["analysis"].group == group]:
            a = row["analysis"]
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"**{a.name}**")
                    st.caption(a.question)
                    st.write(a.output)
                    if a.caution:
                        st.warning(a.caution)
                    need = f"Needs {a.min_batches}+ batches"
                    if a.min_formulations > 1:
                        need += f", {a.min_formulations}+ formulations"
                    if a.needs_dissolution:
                        need += ", dissolution data"
                    if a.needs_response:
                        need += ", a RESPONSE column"
                    st.caption(need)
                with right:
                    if row["available"]:
                        if st.button("Select", key=f"pick_{a.id}"):
                            S.analysis_id = a.id
                            S.result = None
                            st.success(f"Selected: {a.name}")
                    else:
                        st.button("Unavailable", key=f"pick_{a.id}", disabled=True)
                        st.caption(row["reason"])

    if S.get("analysis_id"):
        picked = BY_ID[S.analysis_id]
        st.markdown("---")
        st.success(f"Selected: **{picked.name}**")
        if picked.id in QUICK_IDS:
            st.write("Go to **Quick analysis**.")
        else:
            st.write("Go to **Design diagnostics** and work down the steps in order.")


# ---------------------------------------------------------------------
# 3. Quick analysis
# ---------------------------------------------------------------------
elif choice == "Quick analysis":
    ds = S.ds
    aid = S.get("analysis_id", "")
    picked = BY_ID.get(aid)
    st.subheader(f"3. {picked.name if picked else 'Quick analysis'}")
    if picked and picked.caution:
        st.warning(picked.caution)
    numeric_cols = [c for c in ds.wide.columns if pd.api.types.is_numeric_dtype(ds.wide[c])]

    try:
        if aid == "compare":
            sel = st.multiselect("Batches", ds.batches, default=ds.batches[: min(4, ds.n_batches)])
            fams = st.multiselect("Limit to families", list(FAMILIES))
            if sel:
                st.dataframe(compare_batches(ds, sel, fams or None), use_container_width=True, height=520)

        elif aid == "what_changed":
            c1, c2 = st.columns(2)
            a = c1.selectbox("Batch A", ds.batches, index=0)
            b = c2.selectbox("Batch B", ds.batches, index=min(1, ds.n_batches - 1))
            fam_filter = st.multiselect(
                "Show only these families (recommended: what you deliberately set)",
                list(FAMILIES),
            )
            tbl = what_changed(ds, a, b)
            if fam_filter:
                tbl = tbl[tbl["family"].isin(fam_filter)]
            st.dataframe(tbl, use_container_width=True, height=460)
            st.caption(tbl.attrs.get("caption", ""))

        elif aid == "descriptive":
            st.dataframe(descriptive_summary(ds), use_container_width=True, height=520)

        elif aid == "trend":
            col = st.selectbox("Variable", numeric_cols)
            t = trend_table(ds, col)
            fig, ax = plt.subplots(figsize=(7, 3))
            for f in sorted(pd.unique(t["formulation"])):
                sub = t[t["formulation"] == f]
                ax.plot(sub.index, sub[col], "o-", label=str(f))
            if np.isfinite(t.attrs["mean"]):
                ax.axhline(t.attrs["mean"], color="grey", ls="--", lw=1, label="mean")
            ax.set_xlabel("batch, in manufacturing order")
            ax.set_ylabel(col)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.caption(t.attrs["caption"])

        elif aid == "spec_check":
            st.write("Enter the limits you want to check against.")
            col = st.selectbox("Variable", numeric_cols)
            c1, c2 = st.columns(2)
            lo = c1.number_input("Lower limit", value=float(np.nanmin(ds.wide[col].astype(float))))
            hi = c2.number_input("Upper limit", value=float(np.nanmax(ds.wide[col].astype(float))))
            res = spec_check(ds, {col: (lo, hi)})
            st.dataframe(res, use_container_width=True, hide_index=True, height=380)
            st.caption(res.attrs["caption"])

        elif aid == "f2":
            media = sorted(ds.dissolution["medium"].unique())
            c1, c2, c3 = st.columns(3)
            ref = c1.selectbox("Reference batch", ds.batches, index=0)
            tst = c2.selectbox("Test batch", ds.batches, index=min(1, ds.n_batches - 1))
            med = c3.selectbox("Medium", media)
            r = compare_profiles(ds, ref, tst, med)
            m1, m2, m3 = st.columns(3)
            m1.metric("f2 similarity", f"{r['f2']:.1f}")
            m2.metric("f1 difference", f"{r['f1']:.1f}")
            m3.metric("Time points", r["n_points"])
            (st.success if r["f2"] >= 50 else st.error)(r["verdict"])
            for n in r["notes"]:
                st.warning(n)
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(r["table"]["time_h"], r["table"][ref], "o-", label=ref)
            ax.plot(r["table"]["time_h"], r["table"][tst], "s-", label=tst)
            ax.set_xlabel("hours")
            ax.set_ylabel("% released")
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            st.dataframe(r["table"], use_container_width=True, hide_index=True)

        elif aid == "profile_summary":
            media = sorted(ds.dissolution["medium"].unique())
            med = st.selectbox("Medium", ["all"] + media)
            tbl = profile_summary(ds, None if med == "all" else med)
            st.dataframe(tbl, use_container_width=True, height=460)
            st.caption(
                "Weibull td is the time to about 63% released; MDT is the mean dissolution "
                "time; dissolution efficiency is the area under the curve as a percentage "
                "of complete release over the same window. A blank t80 means that batch "
                "never reached 80% in the time measured, and has not been extrapolated."
            )
        else:
            st.info("Choose one of the describe-and-compare or dissolution analyses first.")
    except Refusal as exc:
        st.error(f"Refused: {exc}")


elif choice == "Design diagnostics":
    st.subheader("4. Design diagnostics")
    st.caption("What this dataset can and cannot answer. Read before modelling anything.")
    from diagnostics import assess_design

    try:
        design = assess_design(S.ds)
    except Refusal as exc:
        st.error(f"Refused: {exc}")
        st.stop()
    S.diagnostics_viewed = True

    for s in design.statements:
        (st.error if "cannot be separated" in s.lower() or "aliased" in s.lower() else st.write)(s)
    for n in design.notes:
        st.info(n)

    st.markdown("### Correlation, clustered into cannot-separate groups")
    order = [m for cluster in design.clusters for m in cluster]
    show = st.slider("Columns to display", 10, max(10, len(order)), min(40, len(order)))
    st.pyplot(_heatmap(design.correlation, order[:show], "Predictor correlation (clustered)"))

    st.markdown("### Inseparable groups")
    if design.inseparable_groups:
        st.dataframe(
            pd.DataFrame(
                {
                    "group": [f"cluster {i+1}" for i in range(len(design.inseparable_groups))],
                    "members": [", ".join(g) for g in design.inseparable_groups],
                }
            ),
            use_container_width=True, hide_index=True,
        )
    else:
        st.write("No pair of predictors exceeded the clustering threshold.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Aliasing with formulation")
        st.dataframe(
            design.formulation_alias_r2.sort_values(ascending=False).head(20).rename("R2"),
            use_container_width=True,
        )
    with c2:
        st.markdown("### Aliasing with manufacturing date")
        st.dataframe(
            design.time_alias_r2.sort_values(ascending=False).head(20).rename("R2"),
            use_container_width=True,
        )

    st.markdown("### Mixture constraints")
    if design.mixture_groups:
        for g in design.mixture_groups:
            st.warning(
                f"{', '.join(g['columns'])} sum to {g['sum_mean']:.2f} (sd {g['sum_sd']:.4f}). "
                "One component will be removed as the dependent remainder before any fit."
            )
    else:
        st.write("No constant-sum component group detected.")

    st.markdown("### Range and replication")
    st.dataframe(design.range_table, use_container_width=True, height=300)


# ---------------------------------------------------------------------
# 3. Response
# ---------------------------------------------------------------------
elif choice == "Response":
    st.subheader("5. Response")
    from response import build_response

    kind = st.radio("Response type", ["Scalar column", "Dissolution profile metric"], horizontal=True)
    method_rsd = st.number_input(
        "Analytical method RSD (%), optional — 0 to skip", min_value=0.0, value=0.0, step=0.1
    )
    kwargs = {}
    if kind == "Scalar column":
        options = S.ds.response_columns() or list(S.ds.wide.columns)
        name = st.selectbox("Response column", options)
        kwargs = {"response_column": name}
    else:
        if S.ds.dissolution is None:
            st.warning("No dissolution table was loaded.")
            st.stop()
        metric = st.selectbox("Profile metric", list(PROFILE_METRICS))
        media = sorted(S.ds.dissolution["medium"].unique())
        medium = None
        if metric != "cross_medium_mdt_spread":
            medium = st.selectbox("Medium", media)
        ref = None
        if metric == "f2_vs_reference":
            ref = st.selectbox("Reference batch", S.ds.batches)
        kwargs = {"profile_metric": metric, "medium": medium, "reference_batch": ref}

    if st.button("Inspect this response"):
        try:
            resp = build_response(
                S.ds,
                kwargs.get("response_column"),
                metric=kwargs.get("profile_metric"),
                medium=kwargs.get("medium"),
                reference_batch=kwargs.get("reference_batch"),
                method_rsd_pct=method_rsd or None,
            )
        except (Refusal, ValueError, KeyError) as exc:
            st.error(f"Refused: {exc}")
            st.stop()
        S.response_choice = {**kwargs, "method_rsd_pct": method_rsd or None,
                             "usable": resp.usable}
        S.result = None

        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Batches with a value", resp.diagnostics["n_observed"])
            st.write(
                {k: v for k, v in resp.diagnostics.items() if k != "variance_components"}
            )
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            vals = resp.values.dropna()
            forms = S.ds.formulations.reindex(vals.index)
            for f in sorted(pd.unique(forms)):
                ax.scatter([f] * (forms == f).sum(), vals[forms == f], s=18)
            ax.set_ylabel(resp.name)
            ax.set_xlabel("formulation")
            fig.tight_layout()
            st.pyplot(fig)

        for s in resp.statements:
            (st.error if not resp.usable else st.info)(s)
        if not resp.usable:
            st.error(
                "This response cannot be attributed. Choosing a different model will not "
                "change that."
            )


# ---------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------
elif choice == "Model":
    st.subheader("6. Model")
    st.caption(
        "Ladder: univariate screen, then regularised linear, then PLS, then non-linear only "
        "if justified. Prefer the simplest that works."
    )
    n = S.ds.n_batches
    from modelling import complexity_cap

    cap = complexity_cap(int(np.floor(n * 0.75)))
    st.info(f"Complexity cap in force: {cap['rule']} (recomputed inside every training fold).")

    model_type = st.selectbox("Model", list(MODEL_TYPES), index=0)
    GUIDE = {
        "elastic_net": "Default. Keeps correlated groups together rather than picking one "
        "member arbitrarily.",
        "lasso": "Sparser than elastic net, but with a confounded pair it keeps one twin "
        "and drops the other with no guarantee which. Use for comparison.",
        "group_lasso": "An inseparable cluster enters or leaves the model as a whole, so a "
        "selected cluster always names every member. Slower: roughly 1 second per fit, so "
        "1000 permutations takes around 15 minutes.",
        "ridge": "Keeps every predictor and shrinks all of them. Selection here means "
        "'inside the complexity cap by absolute effect', not a non-zero coefficient.",
        "pls": "Latent components from correlated predictors. Well suited to process data "
        "where several columns measure the same underlying thing.",
        "scheffe_linear": "Mixture components ONLY, no intercept. Models the simplex "
        "properly instead of deleting a component. Process and material factors are not in "
        "this model and are not tested by it.",
        "scheffe_quadratic": "As above, plus component interaction terms. Needs more batches "
        "than the linear form before it means anything.",
        "gbm": "Non-linear. Only with a reason, and only above about 30 batches.",
        "rf": "Non-linear. Only with a reason, and only above about 30 batches.",
    }
    st.caption(GUIDE.get(model_type, ""))
    if model_type in ("scheffe_linear", "scheffe_quadratic"):
        st.warning(
            "A Scheffe model is fitted on the mixture components alone. It can pass "
            "validation while every process parameter sits outside the model, so a pass "
            "here does not weigh components against process factors or rule them out. "
            "The confound list on each finding still comes from the design diagnostics."
        )
    if model_type == "group_lasso":
        st.info(
            "Group lasso is the slowest option. At 500 permutations expect several minutes; "
            "on the free Streamlit tier consider 200 here and a full run locally."
        )
    if model_type in ("gbm", "rf") and n < 30:
        st.warning(
            f"{model_type} on {n} batches has more capacity than the data. Depth is capped at 3, "
            "and the permutation test is the only thing standing between you and a fiction."
        )
    scheme = st.selectbox("Cross-validation scheme", list(CV_SCHEMES), index=0)
    if scheme == "leave_one_batch_out":
        st.warning(
            "Leave-one-batch-out lets sibling batches of the same formulation train the model "
            "that predicts their neighbour. It answers 'can I predict another batch of a "
            "formulation I have made', not 'can I predict a new formulation'."
        )
    c1, c2 = st.columns(2)
    n_perm = c1.number_input("Permutations", 100, 5000, 500, step=100)
    n_boot = c2.number_input("Bootstrap resamples", 50, 2000, 300, step=50)
    st.caption(
        f"Roughly {n_perm * 0.16 / 60:.1f} minutes on one core: the entire pipeline, "
        "including tuning and variable selection, is refit once per permutation. "
        "Use 1000 for a result you intend to record; the free Streamlit Cloud tier is "
        "single-core and may time out above that."
    )
    allow_missing = st.checkbox(
        "Allow predictors with missing values (fills with the training mean, recorded)", value=False
    )

    if st.button("Fit and validate", type="primary"):
        if S.response_choice is None:
            st.error("Choose a response first.")
            st.stop()
        S.model_configured = True
        with st.spinner(f"Refitting the whole pipeline {n_perm} times for the permutation null..."):
            try:
                S.result = run_attribution(
                    S.ds,
                    S.response_choice.get("response_column"),
                    profile_metric=S.response_choice.get("profile_metric"),
                    medium=S.response_choice.get("medium"),
                    reference_batch=S.response_choice.get("reference_batch"),
                    method_rsd_pct=S.response_choice.get("method_rsd_pct"),
                    model_type=model_type,
                    scheme=scheme,
                    settings=RunSettings(n_permutations=int(n_perm), n_bootstrap=int(n_boot)),
                    allow_missing_predictors=allow_missing,
                )
            except Refusal as exc:
                st.error(f"Refused: {exc}")
                st.stop()
        st.success("Done. Go to Validation.")

    if S.result is not None and len(S.result.screen):
        st.markdown("### Univariate screen")
        st.caption(S.result.screen.attrs.get("caption", ""))
        st.dataframe(S.result.screen.head(25), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------
elif choice == "Validation":
    st.subheader("7. Validation")
    res = S.result
    if res is None:
        st.warning("Fit a model first.")
        st.stop()
    if res.refused:
        for r in res.refusals:
            st.error(r)
        st.stop()

    v = res.validation
    c1, c2, c3 = st.columns(3)
    c1.metric("Cross-validated Q²", f"{v.cv['q2']:.3f}")
    c2.metric("RMSE (held out)", f"{v.cv['rmse']:.3g}")
    c3.metric("Permutation p", f"{v.permutation.get('p_value', float('nan')):.3f}")

    {"supported": st.success, "weak": st.warning, "not_supported": st.error}[v.verdict](
        v.verdict_text
    )
    for s in v.statements[:-1]:
        st.write(s)
    for w in v.warnings + res.warnings:
        st.info(w)

    st.markdown("### Permutation null")
    null = v.permutation.get("null")
    if null is not None and len(null):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(null, bins=30)
        ax.axvline(v.permutation["observed_q2"], color="crimson", lw=2, label="observed")
        ax.axvline(v.permutation["null_p95"], color="grey", ls="--", label="null 95th pct")
        ax.set_xlabel("cross-validated Q² under a shuffled response")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("### Observed against predicted, held-out folds only")
    ovp = v.observed_vs_predicted().dropna()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(ovp["observed"], ovp["predicted_heldout"], s=22)
    lims = [ovp.min().min(), ovp.max().max()]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted (held out)")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### Stability selection")
    st.dataframe(v.stability.table().head(25), use_container_width=True)
    st.markdown("### Folds")
    st.dataframe(v.cv["per_fold"], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# 6. Findings
# ---------------------------------------------------------------------
elif choice == "Findings":
    st.subheader("8. Findings")
    st.error(FINDINGS_BANNER)
    res = S.result
    rep = res.report
    st.markdown(f"**{rep.headline}**")

    for f in rep.findings:
        with st.container(border=True):
            st.markdown(f"**{f.feature}**" + ("  — inseparable cluster" if f.is_cluster else ""))
            st.write(f.sentence(res.response.name))
            if f.confounds:
                st.error("Cannot be separated from: " + ", ".join(f.confounds))
            if not f.stable:
                st.warning("Below the 60% stability threshold. Not a finding.")

    st.markdown("### Dependence and contributions")
    st.caption("Available only because the permutation test passed.")
    members = [m for f in rep.findings for m in f.members]
    if members:
        member = st.selectbox("Factor", members)
        X = res.model_input
        try:
            pdp = partial_dependence(res.model, X[[c for c in X.columns]], member, res.validation)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(pdp["value"], pdp["prediction"])
            ax.set_xlabel(member)
            ax.set_ylabel(f"predicted {res.response.name}")
            fig.tight_layout()
            st.pyplot(fig)
            st.caption(
                "Confounds for this factor: "
                + (", ".join(res.design.confounds_for(member)) or "none identified")
            )
            contrib = linear_contributions(res.model, X, res.validation)
            st.dataframe(contrib.round(3), use_container_width=True, height=250)
        except PermissionError as exc:
            st.error(str(exc))


# ---------------------------------------------------------------------
# 7. Next experiment
# ---------------------------------------------------------------------
elif choice == "Next experiment":
    st.subheader("9. Next experiment")
    res = S.result
    if res.refused:
        for r in res.refusals:
            st.error(r)
        st.stop()
    exp = res.report.experiment if res.report else None
    if exp is None:
        st.info("No confirmatory design could be built from this result.")
    else:
        st.code(exp.text())
        st.dataframe(pd.DataFrame(exp.factors), use_container_width=True, hide_index=True)
        st.caption(
            "This is the smallest design that separates the top association from the factors "
            "it currently travels with."
        )


# ---------------------------------------------------------------------
# 8. Method record
# ---------------------------------------------------------------------
elif choice == "Method record":
    st.subheader("10. Method record")
    if S.result is None:
        st.info("Run an analysis to produce a full record. Decision rules are shown below.")
        from record import DECISION_RULES, REFERENCES

        st.dataframe(
            pd.DataFrame({"rule": list(DECISION_RULES), "definition": list(DECISION_RULES.values())}),
            use_container_width=True, hide_index=True,
        )
        st.dataframe(pd.DataFrame(REFERENCES), use_container_width=True, hide_index=True)
        st.stop()
    md = to_markdown(S.result)
    st.download_button("Download record (Markdown)", md, "method_record.md")
    st.download_button("Download record (JSON)", to_json(S.result), "method_record.json")
    st.markdown(md)
