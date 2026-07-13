"""Streamlit frontend for the ENCHANTED Model 1 dashboard.

This module owns everything the user sees or clicks:
- page styling and header
- authenticated user's top-right identity display
- dashboard tabs
- task queues
- patient list, filters, and table
- patient review form
- audit log and LLM explanation screen
"""

import html

import pandas as pd
import streamlit as st
from pandas.api.types import is_list_like

from app.auth import logout_user, render_auth_gate
from app.backend import (
    BASE_DISPLAY_COLUMNS,
    COLUMN_LABELS,
    SORT_OPTIONS,
    build_review_record,
    call_bedrock_llm,
    filter_worklist,
    load_patient_worklist,
    load_review_log,
    save_review_decision,
)


CATEGORY_OPTIONS = [
    "Green",
    "All Cases",
    "Amber",
    "Red",
    "Pending CM",
    "Pending Clinician",
]

# Options used in the final decision dropdown on the Patient Review tab.
FINAL_DECISION_OPTIONS = [
    "Pending Review",
    "Proceed with Community Hospital Referral",
    "Consider Hospital-at-Home",
    "Continue Acute Hospital Care",
    "Requires Further Clinical Review",
    "Requires Further Nursing Review",
    "Patient / Family Counselling Required",
    "Not Suitable for Transfer",
]


def configure_page():
    """Configure Streamlit page metadata and global CSS overrides."""
    st.set_page_config(page_title="ENCHANTED Model 1", layout="wide")
    # Most CSS below improves readability of Streamlit's generated widgets,
    # especially labels, buttons, dropdowns, radio options, and multiselect tags.
    st.markdown(
        """
        <style>
        .stApp {
            background: #f8fafc;
            color: #111827;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        h1 {
            color: #0f172a;
            font-weight: 800;
            letter-spacing: 0;
        }

        h2, h3 {
            color: #1e293b;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            padding: 18px;
            border-radius: 8px;
            box-shadow: none;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #1f2937 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-weight: 800;
        }

        .stRadio label,
        .stTextInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stTextArea label,
        div[data-testid="stExpander"] summary,
        div[data-testid="stTabs"] button {
            color: #1f2937 !important;
            font-weight: 700 !important;
        }

        .stRadio label,
        .stRadio label p,
        .stRadio label span,
        .stCheckbox label,
        .stCheckbox label p,
        .stCheckbox label span,
        .stMarkdown,
        .stCaptionContainer {
            color: #334155 !important;
            font-weight: 700 !important;
        }

        input,
        textarea,
        input::placeholder,
        textarea::placeholder,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] input {
            color: #111827 !important;
            font-weight: 600 !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        div[data-testid="stDataFrame"] {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px;
            box-shadow: none;
            padding: 8px;
        }

        div[data-testid="stDataFrame"] * {
            color-scheme: light !important;
        }

        div[data-testid="stDataFrame"] canvas,
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataFrame"] [class*="data-grid"],
        div[data-testid="stDataFrame"] [class*="glide"] {
            background-color: #ffffff !important;
            color: #111827 !important;
        }

        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="gridcell"] {
            color: #111827 !important;
            background-color: #ffffff !important;
        }

        .filter-panel {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }

        .filter-panel-title {
            color: #0f172a;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .readable-table-wrap {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            margin-top: 18px;
            overflow-x: auto;
            padding: 0;
        }

        .readable-table {
            border-collapse: collapse;
            width: 100%;
            min-width: 1000px;
            color: #111827;
            font-size: 14px;
        }

        .readable-table th {
            background: #f1f5f9;
            border-bottom: 1px solid #cbd5e1;
            color: #0f172a;
            font-weight: 800;
            padding: 12px 14px;
            text-align: left;
            white-space: nowrap;
        }

        .readable-table td {
            background: #ffffff;
            border-bottom: 1px solid #e2e8f0;
            color: #111827;
            font-weight: 600;
            padding: 11px 14px;
            vertical-align: top;
        }

        .readable-table tr:nth-child(even) td {
            background: #f8fafc;
        }

        .readable-table tr:hover td {
            background: #eef2ff;
        }

        .status-badge {
            border-radius: 6px;
            display: inline-block;
            font-weight: 800;
            padding: 5px 9px;
            white-space: nowrap;
        }

        .status-green {
            background: #dcfce7;
            border: 1px solid #86efac;
            color: #14532d;
        }

        .status-amber {
            background: #fef3c7;
            border: 1px solid #fcd34d;
            color: #78350f;
        }

        .status-red {
            background: #fee2e2;
            border: 1px solid #fca5a5;
            color: #7f1d1d;
        }

        .status-neutral {
            background: #e2e8f0;
            border: 1px solid #cbd5e1;
            color: #0f172a;
        }

        .table-empty {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            color: #475569;
            font-weight: 700;
            margin-top: 18px;
            padding: 16px;
        }

        div[data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }

        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] div,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] label,
        div[data-testid="stExpander"] span {
            color: #111827 !important;
        }

        div[data-testid="stExpander"] summary {
            background: #f1f5f9 !important;
            border-radius: 8px !important;
            padding: 0.5rem 0.75rem !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {
            background: #ffffff !important;
            color: #111827 !important;
        }

        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        ul[role="listbox"] * {
            color: #111827 !important;
        }

        textarea {
            background-color: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid #cbd5e1 !important;
            color: #1f2937 !important;
        }

        input {
            background-color: #ffffff !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 10px;
            border-color: #94a3b8 !important;
            background-color: #ffffff !important;
        }

        div[data-baseweb="select"] svg {
            color: #334155 !important;
            fill: #334155 !important;
        }

        .stMultiSelect div[data-baseweb="tag"],
        .stMultiSelect span[data-baseweb="tag"] {
            background-color: #e2e8f0 !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        .stMultiSelect div[data-baseweb="tag"] span,
        .stMultiSelect span[data-baseweb="tag"] span,
        .stMultiSelect div[data-baseweb="tag"] svg,
        .stMultiSelect span[data-baseweb="tag"] svg {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            gap: 8px;
            border-bottom: 1px solid #cbd5e1;
            margin-bottom: 1rem;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            background: #e2e8f0 !important;
            border: 1px solid #cbd5e1 !important;
            border-bottom: none !important;
            border-radius: 8px 8px 0 0 !important;
            color: #0f172a !important;
            padding: 10px 16px !important;
            font-weight: 700 !important;
        }

        div[data-testid="stTabs"] button[role="tab"] p {
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: #0f172a !important;
            border-color: #0f172a !important;
            color: #ffffff !important;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
            color: #ffffff !important;
        }

        div[data-testid="stTabs"] button[role="tab"]:hover {
            background: #cbd5e1 !important;
            color: #0f172a !important;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]:hover {
            background: #1e293b !important;
            color: #ffffff !important;
        }

        .stRadio div[role="radiogroup"] {
            gap: 0.5rem 1.2rem;
        }

        .stRadio div[role="radiogroup"] label {
            margin-right: 0.75rem;
        }

        div.stButton > button {
            background: #0f172a !important;
            color: #ffffff !important;
            border: 1px solid #0f172a !important;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 800;
            box-shadow: none;
        }

        div.stButton > button p,
        div.stButton > button span,
        div.stButton > button div {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        div.stButton > button:hover {
            background: #334155 !important;
            color: #ffffff !important;
            border: 1px solid #334155 !important;
        }

        div.stButton > button:focus {
            color: #ffffff !important;
            border: 2px solid #38bdf8 !important;
            outline: none !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            background: #0f172a !important;
            color: #ffffff !important;
            border: 1px solid #0f172a !important;
            font-weight: 800 !important;
        }

        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stFormSubmitButton"] button span {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        hr {
            border: none;
            height: 1px;
            background: #cbd5e1;
            margin: 1.5rem 0;
        }

        [data-testid="stDataFrame"] div[data-baseweb="tag"],
        [data-testid="stDataFrame"] div[data-baseweb="tag"] span,
        [data-testid="stDataFrame"] span[data-baseweb="tag"] {
            font-weight: 700 !important;
            color: #000000 !important;
        }

        .top-bar {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 24px;
        }

        .user-chip {
            min-width: 220px;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: none;
            text-align: right;
        }

        .user-chip-name {
            color: #1f2937;
            font-size: 14px;
            font-weight: 700;
        }

        .user-chip-role {
            color: #0f172a;
            font-size: 13px;
            font-weight: 700;
            margin-top: 2px;
        }

        .user-chip-id {
            color: #64748b;
            font-size: 12px;
            margin-top: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(user_profile):
    """Render the dashboard title and logged-in user's designation chip."""
    st.markdown(
        f"""
        <div class="top-bar">
            <div>
                <h1>ENCHANTED Model 1: Right-Siting Decision Support Dashboard</h1>
                <div style="font-size: 22px; color: #334155; font-weight: 700;">
                    Rule-Based Screening, AI Risk Stratification and Right-Siting Review Support
                </div>
            </div>
            <div class="user-chip">
                <div class="user-chip-name">{user_profile["name"]}</div>
                <div class="user-chip-role">{user_profile["designation"]}</div>
                <div class="user-chip-id">{user_profile["hospital_id"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "For demonstration using sample data. Final right-siting, referral and transfer decisions remain with the clinical care team."
    )
    _, logout_col = st.columns([0.86, 0.14])
    if logout_col.button("Logout"):
        logout_user()

    st.markdown(
        """
        <div style="
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-left: 6px solid #0f172a;
            padding: 18px 22px;
            border-radius: 8px;
            box-shadow: none;
            margin-bottom: 20px;
        ">
            <div style="font-size: 18px; font-weight: 700; color: #0f172a;">
                Acute-to-Community Hospital / Hospital-at-Home Right-Siting Support
            </div>
            <div style="font-size: 14px; color: #475569; margin-top: 6px;">
            This dashboard combines rule-based clinical screening, AI-supported risk stratification,
            service suitability, nursing assessment and acceptance considerations to assist case managers
            and clinicians in reviewing the appropriate care pathway.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap;">
            <span style="background:#fee2e2; color:#7f1d1d; border:1px solid #fecaca; padding:8px 14px; border-radius:8px; font-weight:700;">
                Red: Rule-Based Exclusion
            </span>
            <span style="background:#fef3c7; color:#78350f; border:1px solid #fde68a; padding:8px 14px; border-radius:8px; font-weight:700;">
                Amber: Clinical Review Required
            </span>
            <span style="background:#dcfce7; color:#14532d; border:1px solid #bbf7d0; padding:8px 14px; border-radius:8px; font-weight:700;">
                Green: Potential Candidate
            </span>
            <span style="background:#e2e8f0; color:#0f172a; border:1px solid #cbd5e1; padding:8px 14px; border-radius:8px; font-weight:700;">
                AI: Risk Stratification
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def colour_rule_category(value):
    """Return cell styling for screening and recommendation status values."""
    if value == "Green - Potential Candidate":
        return "background-color: #d4edda; color: #155724;"
    if value == "Amber - Review Required":
        return "background-color: #fff3cd; color: #856404;"
    if value == "Red - No-Go":
        return "background-color: #f8d7da; color: #721c24;"
    if value == "Community Hospital referral review":
        return "background-color: #d4edda; color: #155724; font-weight: bold;"
    if value == "Further clinical review before CH referral":
        return "background-color: #fff3cd; color: #856404; font-weight: bold;"
    if value == "Pending review":
        return "background-color: #e2e3e5; color: #383d41; font-weight: bold;"
    return ""


def role_scope(shortlisted, user_profile):
    """Return only the cases that should be visible to the logged-in user."""
    selected_role = user_profile["designation"]

    if selected_role == "Case Manager":
        assigned_case_manager = user_profile.get("assigned_case_manager")
        return shortlisted[shortlisted["assigned_case_manager"] == assigned_case_manager]

    if selected_role == "Clinician":
        return shortlisted[shortlisted["workflow_status"] == "Pending Clinician"]

    return shortlisted


def render_metrics(role_scoped):
    """Render summary metrics for the cases visible to the current user."""
    metric_cols = st.columns(5)
    metric_cols[0].metric("Visible Cases", len(role_scoped))
    metric_cols[1].metric(
        "Green / Candidate",
        (role_scoped["rule_category"] == "Green - Potential Candidate").sum(),
    )
    metric_cols[2].metric(
        "Amber / Review",
        (role_scoped["rule_category"] == "Amber - Review Required").sum(),
    )
    metric_cols[3].metric("Red / No-Go", (role_scoped["rule_category"] == "Red - No-Go").sum())
    metric_cols[4].metric(
        "Clinician Queue",
        (role_scoped["workflow_status"] == "Pending Clinician").sum(),
    )

    st.subheader("Review Pathway Workload Summary")
    st.caption(
        "This summarizes where the currently visible cases should be reviewed next. "
        "It is workflow guidance for triage and workload planning, not a final transfer decision."
    )
    pathway_cols = st.columns(5)
    pathway_cols[0].metric(
        "Community Hospital Review",
        (role_scoped["right_siting_recommendation"] == "Community Hospital review").sum(),
    )
    pathway_cols[1].metric(
        "Hospital-at-Home Review",
        (role_scoped["right_siting_recommendation"] == "Hospital-at-Home review").sum(),
    )
    pathway_cols[2].metric(
        "Remain in Acute Care",
        (
            role_scoped["right_siting_recommendation"]
            == "Continue Acute Hospital care"
        ).sum(),
    )
    pathway_cols[3].metric(
        "Needs Further Review",
        (
            role_scoped["right_siting_recommendation"]
            == "Further clinical / nursing review required"
        ).sum(),
    )
    pathway_cols[4].metric(
        "Case Manager Queue",
        (role_scoped["workflow_status"] == "Pending CM").sum(),
    )
    st.info(
        "Use this section to understand the next review destination: Community Hospital, "
        "Hospital-at-Home, acute care continuation, or further clinical/nursing review."
    )


def split_tasks_for_user(role_scoped, user_profile):
    """Split visible cases into assigned, ongoing, and past-reviewed groups."""
    review_log = load_review_log()

    if review_log.empty:
        reviewed_patient_ids = set()
    else:
        if user_profile["designation"] == "Case Manager":
            if "assigned_case_manager" in review_log.columns:
                matching_reviews = review_log[
                    review_log["assigned_case_manager"]
                    == user_profile.get("assigned_case_manager")
                ]
            else:
                matching_reviews = review_log.iloc[0:0]
        else:
            if "dashboard_role" in review_log.columns:
                matching_reviews = review_log[
                    review_log["dashboard_role"] == user_profile["designation"]
                ]
            else:
                matching_reviews = review_log.iloc[0:0]

        if "patient_id" in matching_reviews.columns:
            reviewed_patient_ids = set(matching_reviews["patient_id"].dropna())
        else:
            reviewed_patient_ids = set()

    past = role_scoped[role_scoped["patient_id"].isin(reviewed_patient_ids)]
    active = role_scoped[~role_scoped["patient_id"].isin(reviewed_patient_ids)]

    if user_profile["designation"] == "Case Manager":
        assigned = active[active["workflow_status"] == "Pending CM"]
    elif user_profile["designation"] == "Clinician":
        assigned = active[active["workflow_status"] == "Pending Clinician"]
    else:
        assigned = active[
            active["workflow_status"].isin(["Pending CM", "Pending Clinician"])
        ]

    ongoing = active[~active["patient_id"].isin(assigned["patient_id"])]

    return assigned, ongoing, past


def render_task_table(data, empty_message):
    """Render a compact task table using the same style as the patient list."""
    task_columns = [
        "patient_id",
        "encounter_id",
        "ward",
        "age",
        "rule_category",
        "risk_band",
        "workflow_status",
        "right_siting_recommendation",
    ]
    visible_columns = [col for col in task_columns if col in data.columns]

    if data.empty:
        st.info(empty_message)
        return

    render_readable_table(data, visible_columns)


def render_task_sections(role_scoped, user_profile):
    """Render the Overview & Tasks tab content."""
    assigned, ongoing, past = split_tasks_for_user(role_scoped, user_profile)

    st.subheader("Your Tasks")
    task_metric_cols = st.columns(3)
    task_metric_cols[0].metric("Tasks Assigned to You", len(assigned))
    task_metric_cols[1].metric("Ongoing", len(ongoing))
    task_metric_cols[2].metric("Past", len(past))

    assigned_tab, ongoing_tab, past_tab = st.tabs(
        ["Tasks assigned to you", "Ongoing", "Past"]
    )

    with assigned_tab:
        render_task_table(assigned, "No assigned tasks for your current designation.")

    with ongoing_tab:
        render_task_table(ongoing, "No ongoing cases outside your assigned task queue.")

    with past_tab:
        render_task_table(past, "No past reviewed cases yet.")


def render_worklist_controls(shortlisted, role_scoped):
    """Render search, sort, column, category, and advanced-filter controls."""
    control_cols = st.columns([1.4, 1.2, 1, 1])
    selected_category = control_cols[0].radio(
        "Case category",
        CATEGORY_OPTIONS,
        horizontal=True,
    )
    search_query = control_cols[1].text_input(
        "Search",
        placeholder="Patient, encounter, ward, specialty",
    )
    sort_label = control_cols[2].selectbox(
        "Sort by",
        list(SORT_OPTIONS.keys()),
        index=1,
    )
    sort_direction = control_cols[3].selectbox(
        "Sort direction",
        ["Descending", "Ascending"],
    )

    available_columns = [col for col in COLUMN_LABELS.keys() if col in shortlisted.columns]
    selected_columns = st.multiselect(
        "Columns",
        available_columns,
        default=[col for col in BASE_DISPLAY_COLUMNS if col in available_columns],
        format_func=lambda col: COLUMN_LABELS[col],
    )

    st.markdown(
        """
        <div class="filter-panel">
            <div class="filter-panel-title">Advanced filters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filter_cols = st.columns(5)
    ward_filter = filter_cols[0].multiselect(
        "Ward / Floor",
        sorted(role_scoped["ward"].dropna().unique()),
    )
    status_filter = filter_cols[1].multiselect(
        "Status",
        sorted(role_scoped["workflow_status"].dropna().unique()),
    )
    risk_filter = filter_cols[2].multiselect(
        "Risk Band",
        sorted(role_scoped["risk_band"].dropna().unique()),
    )
    specialty_filter = filter_cols[3].multiselect(
        "Specialty",
        sorted(role_scoped["specialty"].dropna().unique()),
    )
    age_filter = filter_cols[4].slider(
        "Age Range",
        int(shortlisted["age"].min()),
        int(shortlisted["age"].max()),
        (int(shortlisted["age"].min()), int(shortlisted["age"].max())),
    )

    filtered_worklist = filter_worklist(
        role_scoped,
        selected_category,
        search_query,
        ward_filter,
        status_filter,
        risk_filter,
        age_filter,
        specialty_filter,
        sort_label,
        sort_direction == "Ascending",
    )

    return filtered_worklist, selected_columns


def format_table_value(value):
    """Convert dataframe cell values into compact display text for HTML tables."""
    if isinstance(value, float):
        return f"{value:.2f}"

    if is_list_like(value) and not isinstance(value, str):
        values = [str(item) for item in value if str(item)]
        return ", ".join(values) if values else "-"

    if pd.isna(value):
        return "-"

    return str(value)


def clinical_screening_badge(value):
    """Return HTML for a color-coded clinical screening badge."""
    value_text = format_table_value(value)

    if value_text == "Green - Potential Candidate":
        badge_class = "status-green"
    elif value_text == "Amber - Review Required":
        badge_class = "status-amber"
    elif value_text == "Red - No-Go":
        badge_class = "status-red"
    else:
        badge_class = "status-neutral"

    return (
        f'<span class="status-badge {badge_class}">'
        f"{html.escape(value_text)}"
        "</span>"
    )


def dataframe_to_readable_html(data):
    """Build table HTML manually so selected columns can use custom badges."""
    headers = [html.escape(str(column)) for column in data.columns]
    header_html = "".join(f"<th>{header}</th>" for header in headers)

    rows = []
    for _, row in data.iterrows():
        cells = []
        for column, value in row.items():
            if column == "Clinical Screening":
                cell_value = clinical_screening_badge(value)
            else:
                cell_value = html.escape(format_table_value(value))
            cells.append(f"<td>{cell_value}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<table class="readable-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def render_readable_table(data, selected_columns):
    """Render a light HTML table instead of Streamlit's dark canvas dataframe."""
    if data.empty:
        st.markdown(
            '<div class="table-empty">No patients match the current filters.</div>',
            unsafe_allow_html=True,
        )
        return

    display_df = data[selected_columns].copy()
    display_df = display_df.rename(columns=COLUMN_LABELS)
    table_html = dataframe_to_readable_html(display_df)

    st.markdown(
        f'<div class="readable-table-wrap">{table_html}</div>',
        unsafe_allow_html=True,
    )


def render_worklist_table(filtered_worklist, role_scoped, selected_columns):
    """Render the interactive patient list table with selected columns."""
    if selected_columns:
        render_readable_table(filtered_worklist, selected_columns)
    else:
        st.warning("Select at least one column to display.")

    st.caption(f"Showing {len(filtered_worklist)} of {len(role_scoped)} role-scoped cases.")


def render_patient_list_tab(shortlisted, role_scoped):
    """Render the Patient List tab and return its filtered dataframe."""
    st.subheader("All Patient List")
    st.caption(
        "Use the case category switch, search, sorting, column selector, and advanced filters to review the full list visible to your designation."
    )
    render_metrics(role_scoped)
    st.divider()

    filtered_worklist, selected_columns = render_worklist_controls(
        shortlisted,
        role_scoped,
    )
    render_worklist_table(filtered_worklist, role_scoped, selected_columns)

    return filtered_worklist


def select_patient_for_review(shortlisted, role_scoped):
    """Render patient selection for the review screen and return the row."""
    st.subheader("Select Patient for Detail Review")
    st.caption(
        "Choose a patient from your visible case list, then complete the review workflow below."
    )

    patient_options = role_scoped["patient_id"].tolist()

    if not patient_options:
        st.info("No patients are visible for your current designation.")
        st.stop()

    selected_patient = st.selectbox(
        "Patient",
        patient_options,
        format_func=lambda patient_id: (
            f"{patient_id} - "
            f"{shortlisted.loc[shortlisted['patient_id'] == patient_id, 'encounter_id'].iloc[0]}"
        ),
    )

    return shortlisted[shortlisted["patient_id"] == selected_patient].iloc[0]


def render_patient_review_tab(shortlisted, role_scoped, user_profile):
    """Render patient-level details and the final review decision form."""
    patient_row = select_patient_for_review(shortlisted, role_scoped)

    st.write("### Screening Output")
    st.write(f"**Rule-based category:** {patient_row['rule_category']}")
    st.write(f"**Workflow status:** {patient_row['workflow_status']}")
    st.write(f"**Assigned case manager:** {patient_row['assigned_case_manager']}")
    st.write(f"**Red flags:** {patient_row['red_flags']}")
    st.write(f"**Amber flags:** {patient_row['amber_flags']}")

    if pd.notna(patient_row["risk_score"]):
        st.write(f"**Predictive risk score:** {patient_row['risk_score']:.2f}")
    else:
        st.write("**Predictive risk score:** Not applicable")

    st.write(f"**Predictive risk band:** {patient_row['risk_band']}")
    st.write(f"**AI-supported recommendation:** *{patient_row['ai_recommendation']}*")

    st.write("### Service Suitability")
    st.write(f"**Service need:** {patient_row['service_need']}")
    st.write(f"**Service suitability:** {patient_row['service_suitability']}")

    st.write("### Nursing / Operational Assessment")
    st.write(f"**Nursing status:** {patient_row['nursing_status']}")
    st.write(f"**Nursing flags:** {patient_row['nursing_flags']}")

    st.write("### Acceptance / Counselling Assessment")
    st.write(
        f"**Patient acceptance likelihood:** {patient_row['patient_acceptance_likelihood']}"
    )
    st.write(f"**Counselling required:** {patient_row['counselling_required']}")

    st.write("### Right-Siting Recommendation")
    st.write(f"**Recommended care setting:** {patient_row['right_siting_recommendation']}")
    st.write(f"**AI-supported recommendation:** *{patient_row['ai_recommendation']}*")

    final_decision = st.selectbox(
        "Final right-siting decision",
        FINAL_DECISION_OPTIONS,
    )
    review_comments = st.text_area("Review comments / override reason")

    st.info(
        "The AI model provides decision support only. "
        "Final referral decisions remain with the case manager / clinical team."
    )

    if st.button("Submit Review Decision"):
        review_record = build_review_record(
            patient_row,
            user_profile["designation"],
            final_decision,
            review_comments,
        )
        save_review_decision(review_record)
        st.success("Review decision submitted and saved to audit log.")


def render_audit_and_llm_tab(shortlisted, role_scoped):
    """Render submitted review history and optional Bedrock explanation support."""
    st.subheader("Audit Log")
    review_log = load_review_log()

    if review_log.empty:
        st.info("No review decisions have been submitted yet.")
    else:
        st.dataframe(review_log, width="stretch")

    st.divider()
    st.subheader("LLM Explanation Support")

    patient_options = role_scoped["patient_id"].tolist()

    if not patient_options:
        st.info("No patients are visible for your current designation.")
        return

    selected_patient = st.selectbox(
        "Patient for LLM prompt",
        patient_options,
        key="llm_patient_selector",
    )
    patient_row = shortlisted[shortlisted["patient_id"] == selected_patient].iloc[0]

    st.text_area(
        "Prompt that would be sent to the LLM explanation layer",
        patient_row["llm_prompt"],
        height=400,
    )

    if st.button("Generate LLM Explanation"):
        with st.spinner("Generating explanation..."):
            llm_output = call_bedrock_llm(patient_row["llm_prompt"])

        st.markdown(llm_output)


def render_dashboard():
    """Main frontend orchestrator called by classification.py."""
    configure_page()
    user_profile = render_auth_gate()
    shortlisted = load_patient_worklist()

    render_header(user_profile)
    role_scoped = role_scope(shortlisted, user_profile)

    overview_tab, patient_list_tab, review_tab, audit_tab = st.tabs(
        [
            "Overview & Tasks",
            "Patient List",
            "Patient Review",
            "Audit & LLM",
        ]
    )

    with overview_tab:
        render_task_sections(role_scoped, user_profile)

    with patient_list_tab:
        render_patient_list_tab(shortlisted, role_scoped)

    with review_tab:
        render_patient_review_tab(shortlisted, role_scoped, user_profile)

    with audit_tab:
        render_audit_and_llm_tab(shortlisted, role_scoped)
