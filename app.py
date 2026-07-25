"""
app.py
학교안전사고 예방 의사결정 지원 시스템 (Streamlit)

실행:  streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Secrets 브리지 -----------------------------------------------------------
# Streamlit Community Cloud 의 Secrets(st.secrets)를 환경변수로 옮겨,
# 로컬(환경변수)과 클라우드(Secrets) 어디서나 advisor 가 동일하게 동작하게 한다.
for _k in ("GITHUB_TOKEN", "GITHUB_MODEL", "GEMINI_API_KEY",
           "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    try:
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass  # secrets 파일이 없어도 로컬 환경변수로 동작

from backend import (
    SCHOOL_LEVELS, PLACES, ACCIDENT_TYPES, RISK_GRADES,
    analyze_accident, get_top10_priority,
)
from advisor import generate_safety_advice

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="학교안전사고 예방 의사결정 지원 시스템",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- 공공기관 스타일 테마 -----------------------------------------------------
st.markdown(
    """
    <style>
    :root{
      --navy:#10243e; --blue:#2563eb; --blue-d:#1e5aa0; --sky:#eaf2ff;
      --bg:#f4f7fb; --text:#172033; --muted:#657086; --line:#dbe3ef;
      --green:#0f9d72; --amber:#d97706; --red:#dc2626;
      --shadow:0 12px 30px rgba(16,36,62,.09);
    }
    html, body, [class*="css"]{
      font-family:"Pretendard","Noto Sans KR","Apple SD Gothic Neo",
                  -apple-system,BlinkMacSystemFont,sans-serif;
      color:var(--text);
    }
    .stApp{ background:var(--bg); }
    .block-container{ max-width:1240px; padding-top:1.1rem; padding-bottom:4rem; }

    /* 상단 헤더 배너 (네이비 그라데이션) */
    .app-header{
      background:linear-gradient(135deg,#0b1f38,#173f70); color:#fff;
      border-radius:20px; padding:30px 30px 34px; margin:2px 0 22px;
      box-shadow:var(--shadow);
    }
    .app-header .eyebrow{ font-size:13px; letter-spacing:.12em; font-weight:800; color:#a9ccff; }
    .app-header .title{ font-size:30px; font-weight:850; margin:8px 0 6px; letter-spacing:-.01em; }
    .app-header .subtitle{ max-width:840px; color:#d9e7fa; line-height:1.65; font-size:15px; }

    /* 섹션 제목 + 번호 뱃지 */
    .section-title{
      display:flex; align-items:center; gap:10px; font-size:18px;
      font-weight:800; color:var(--navy); margin:8px 0 6px;
    }
    .section-title .badge{
      width:28px; height:28px; border-radius:9px; display:grid; place-items:center;
      background:var(--sky); color:var(--blue); font-size:14px; font-weight:800; flex:none;
    }
    h2, .stMarkdown h2{ color:var(--navy); }

    /* 분석 대상 태그 */
    .target-tag{
      display:inline-block; background:var(--sky); border:1px solid #cfe0fb;
      border-radius:10px; padding:9px 14px; font-size:14.5px; font-weight:750;
      color:var(--blue-d); margin:2px 0 14px;
    }

    /* 버튼 */
    .stButton > button{
      background:var(--blue); color:#fff; border:none; border-radius:10px;
      font-weight:750; padding:10px 16px;
    }
    .stButton > button:hover{ background:var(--blue-d); color:#fff; }

    /* selectbox */
    div[data-baseweb="select"] > div{ border-radius:11px; border-color:#cbd6e5; }

    /* 사이드바 */
    section[data-testid="stSidebar"]{ background:#fff; border-right:1px solid var(--line); }
    section[data-testid="stSidebar"] h2{ font-size:17px; color:var(--navy); }

    /* st.metric 카드화 (SHAP 요약) */
    div[data-testid="stMetric"]{
      background:#fff; border:1px solid var(--line); border-radius:14px;
      padding:12px 14px; box-shadow:0 6px 16px rgba(16,36,62,.05);
    }

    /* 구분선 */
    hr{ margin:1.1rem 0; border:none; border-top:1px solid var(--line); }

    /* 표·그래프 가로 스크롤 */
    [data-testid="stDataFrame"], .stPlotlyChart{ overflow-x:auto; }

    /* 모바일 */
    @media (max-width: 640px){
      .block-container{ padding:1rem .8rem 3rem !important; }
      .app-header{ padding:22px 20px 26px; }
      .app-header .title{ font-size:23px; }
      .section-title{ font-size:16px; }
      div[data-testid="stHorizontalBlock"]{ flex-wrap:wrap !important; }
      div[data-testid="stHorizontalBlock"] > div[data-testid="column"]{
        flex:1 1 100% !important; width:100% !important; min-width:100% !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 위험등급 색상 — 신호등 체계(HTML 팔레트: red→amber→green)
GRADE_COLOR = {
    "1등급_초고위험": "#dc2626",
    "2등급_심각위험": "#e5484d",
    "3등급_고위험": "#d97706",
    "4등급_중점관리": "#ca8a04",
    "5등급_일반위험": "#16a34a",
    "6등급_저위험": "#0f9d72",
}

# HTML 스타일 섹션 제목(번호 뱃지) 렌더러
def section_title(num, text):
    st.markdown(
        f'<div class="section-title"><span class="badge">{num}</span>{text}</div>',
        unsafe_allow_html=True,
    )

# plotly 공통 테마(밝은 카드 배경에 맞춤)
def _theme(fig, height, top=52):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=top, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Pretendard, Noto Sans KR, sans-serif",
                  color="#172033", size=13),
    )
    return fig

st.markdown(
    """
    <div class="app-header">
      <div class="eyebrow">SCHOOL SAFETY DECISION SUPPORT</div>
      <div class="title">🏫 학교안전사고 예방 의사결정 지원 시스템</div>
      <div class="subtitle">학교급·사고장소·사고형태를 선택하면 위험등급 분류 · 예상 보상금 예측 ·
        CRITIC-TOPSIS 우선순위 · SHAP 해석을 한 화면에서 제공합니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 입력 영역
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("사고 조건 입력")
    school = st.selectbox("학교급", SCHOOL_LEVELS, index=1)
    place = st.selectbox("사고장소", PLACES, index=0)
    atype = st.selectbox("사고형태", ACCIDENT_TYPES, index=0)
    run = st.button("예방 우선순위 분석", type="primary", use_container_width=True)
    st.divider()
    st.caption("입력 항목 : 학교급 · 사고장소 · 사고형태")

if not run and "result" not in st.session_state:
    st.markdown(
        '<div style="background:#fff;border:1px solid #dbe3ef;border-radius:16px;'
        'padding:22px 24px;font-size:15px;color:#172033;'
        'box-shadow:0 8px 20px rgba(16,36,62,.05);">'
        '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        'background:#0f9d72;margin-right:8px;"></span>'
        '왼쪽 <b>사고 조건 입력</b>에서 학교급 · 사고장소 · 사고형태를 선택한 뒤 '
        '<b>[예방 우선순위 분석]</b> 버튼을 눌러 주십시오.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if run:
    st.session_state["result"] = analyze_accident(school, place, atype)
    st.session_state["ctx"] = {"학교급": school, "사고장소": place, "사고형태": atype}
    st.session_state.pop("advice", None)  # 새 분석 시 조언 초기화

r = st.session_state["result"]
ctx_in = st.session_state["ctx"]
grade = r["위험등급"]
grade_color = GRADE_COLOR.get(grade, "#7f8c8d")

st.markdown(
    f'<div class="target-tag">분석 대상 &nbsp;|&nbsp; '
    f'{ctx_in["학교급"]} · {ctx_in["사고장소"]} · {ctx_in["사고형태"]}</div>',
    unsafe_allow_html=True,
)

# ===========================================================================
# 1. 결과 카드 4종
# ===========================================================================
c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1])


def hero_card(col, label, value, sub, accent):
    col.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#14345c,#1e5aa0);color:#fff;
                    border-top:4px solid {accent};border-radius:17px;
                    padding:18px 19px;min-height:120px;margin-bottom:10px;
                    box-shadow:0 10px 24px rgba(16,36,62,.16);">
          <div style="font-size:12px;font-weight:750;color:#bdd7f6;
                      text-transform:uppercase;letter-spacing:.06em;">{label}</div>
          <div style="font-size:clamp(19px,4.4vw,23px);font-weight:850;
                      margin-top:8px;line-height:1.25;">{value}</div>
          <div style="font-size:12px;color:#d3e4f8;margin-top:9px;line-height:1.45;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(col, label, value, sub, accent):
    col.markdown(
        f"""
        <div style="background:#fff;border:1px solid #dbe3ef;border-radius:17px;
                    padding:18px 19px;min-height:120px;margin-bottom:10px;
                    box-shadow:0 8px 20px rgba(16,36,62,.05);">
          <div style="font-size:12px;font-weight:750;color:#657086;
                      text-transform:uppercase;letter-spacing:.06em;">{label}</div>
          <div style="font-size:clamp(20px,4.6vw,24px);font-weight:850;color:{accent};
                      margin-top:9px;line-height:1.2;">{value}</div>
          <div style="font-size:12px;color:#657086;margin-top:6px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


hero_card(c1, "위험등급", grade.replace("_", " "),
          f"6단계 중 {r['위험등급_index'] + 1}단계 · 상대적 개선 우선순위", grade_color)
metric_card(c2, "예상 보상금", f"{r['예상보상금']:,}원",
            "사고 1건당 예측 총보상액", "#1e5aa0")
metric_card(c3, "사고 빈도", f"{r['사고빈도']:,}건",
            "연간 발생 추정 건수", "#0f9d72")
metric_card(c4, "예방 우선순위", f"{r['예방우선순위']}위",
            f"전체 {r['전체_사고유형수']}개 사고유형 중", "#dc2626")

st.divider()

# ===========================================================================
# 2. 위험도 시각화 (위험확률 · TOPSIS · 예상보상금 · 사고빈도 · SHAP)
# ===========================================================================
section_title(1, "위험도 시각화")
st.caption(
    "왼쪽은 이 사고유형의 종합 예방 우선순위 점수(TOPSIS), "
    "오른쪽은 그 점수를 만들어 낸 세부 지표(예방 우선순위 선정 근거)입니다."
)
norm = r["지표_정규화"]


def gauge(title, value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"color": "#10243e"}},
        title={"text": title, "font": {"size": 16, "color": "#10243e"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#eaf2ff"},
                {"range": [40, 70], "color": "#fff3df"},
                {"range": [70, 100], "color": "#fdeaea"},
            ],
        },
    ))
    return _theme(fig, height=250)


g1, g2 = st.columns([1, 1.1])

with g1:
    # TOPSIS 근접계수 게이지 + 설명
    st.plotly_chart(gauge("TOPSIS 근접계수", norm["TOPSIS"], "#1e5aa0"),
                    use_container_width=True)
    st.markdown(
        "**TOPSIS 근접계수** : 위험확률·예상보상금·사고빈도 세 지표를 결합하여, "
        "'가장 시급히 관리해야 할 이상적 사고유형'에 얼마나 가까운지를 0~1로 나타낸 "
        "종합 우선순위 점수입니다. 1에 가까울수록 예방 우선순위가 높습니다."
    )

with g2:
    # 예방 우선순위 선정 근거 — 위험확률·예상보상금·사고빈도 가로 막대
    bar = go.Figure()
    metrics = ["위험확률", "예상보상금", "사고빈도"]
    vals = [norm["위험확률"], norm["예상보상금"], norm["사고빈도"]]
    colors = ["#dc2626", "#d97706", "#2563eb"]
    bar.add_trace(go.Bar(
        x=[v * 100 for v in vals], y=metrics, orientation="h",
        marker_color=colors,
        text=[f"{v * 100:.0f}%" for v in vals], textposition="outside",
    ))
    bar.update_layout(
        title="예방 우선순위 선정 근거",
        xaxis=dict(range=[0, 115], title="지표 수준 (0~100%)",
                   gridcolor="#e6ecf5", zerolinecolor="#e6ecf5"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(_theme(bar, height=300), use_container_width=True)
    st.caption("위 TOPSIS 점수를 구성하는 세 지표입니다. "
               "각 막대가 높을수록 그 지표가 예방 우선순위를 끌어올린 것입니다.")

st.divider()

# ===========================================================================
# 3. SHAP 분석 (기여 요인 시각화 + 중요 요인 요약 통합)
# ===========================================================================
section_title(2, "SHAP 분석 — 무엇이 이 사고의 위험을 높였나")
st.caption(
    "SHAP은 위험등급 예측에 각 입력 요인(학교급·사고장소·사고형태)이 "
    "**얼마나, 어느 방향으로 기여했는지**를 수치로 설명합니다. "
    "값이 클수록(+) 위험등급을 더 크게 끌어올린 요인입니다."
)

shap_df = pd.DataFrame(r["shap"])
top_shap = shap_df.iloc[0]

sh1, sh2 = st.columns([1.6, 1])

with sh1:
    shap_max = float(shap_df["기여도"].max())
    sfig = go.Figure(go.Bar(
        x=shap_df["기여도"], y=shap_df["요인"], orientation="h",
        marker_color="#1e5aa0", width=0.66,
        text=[f"+{v:.3f}" for v in shap_df["기여도"]],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=15),
    ))
    sfig.update_layout(
        title="요인별 위험등급 상승 기여도 (SHAP value)",
        xaxis=dict(title="위험등급 상승 기여도 (클수록 위험 ↑)",
                   range=[0, shap_max * 1.22],
                   gridcolor="#e6ecf5", zerolinecolor="#e6ecf5"),
        yaxis=dict(autorange="reversed", automargin=True,
                   tickfont=dict(size=14)),
        bargap=0.3,
    )
    st.plotly_chart(_theme(sfig, height=400, top=55), use_container_width=True)

with sh2:
    st.markdown("**요인별 기여도**")
    for _, row in shap_df.iterrows():
        st.metric(row["요인"], f"+{row['기여도']:.3f}", "위험 상승 기여")

st.markdown(
    f'<div style="background:#f5f9ff;border:1px solid #cdddf5;border-left:4px solid #2563eb;'
    f'border-radius:0 12px 12px 0;padding:14px 18px;font-size:15px;'
    f'line-height:1.7;color:#172033;">'
    f'가장 큰 위험 상승 요인은 <b>{top_shap["요인"]}</b> '
    f'(기여도 +{top_shap["기여도"]:.3f})으로, 위험등급 예측을 높이는 방향으로 '
    f'가장 강하게 작용했습니다. 이 요인을 우선적으로 관리하면 위험을 가장 효과적으로 '
    f'낮출 수 있습니다.</div>',
    unsafe_allow_html=True,
)

st.divider()

# ===========================================================================
# 4. 예방 우선순위 선정 이유 (문장)
# ===========================================================================
section_title(3, "예방 우선순위 선정 이유")


def priority_reason(res: dict) -> str:
    n = res["지표_정규화"]
    def lvl(v):
        return "매우 높음" if v >= 0.75 else "높음" if v >= 0.55 \
            else "보통" if v >= 0.35 else "낮음"
    key_driver = max(
        [("위험 확률", n["위험확률"]), ("예상 보상금", n["예상보상금"]),
         ("사고 빈도", n["사고빈도"])],
        key=lambda x: x[1],
    )[0]
    return (
        f"이 사고유형은 위험 확률이 <b>{res['위험확률']*100:.0f}%({lvl(n['위험확률'])})</b>, "
        f"사고 빈도가 <b>{res['사고빈도']:,}건({lvl(n['사고빈도'])})</b>, "
        f"예상 보상금이 <b>{res['예상보상금']:,}원({lvl(n['예상보상금'])})</b> 수준입니다. "
        f"CRITIC 가중치로 세 지표를 결합한 <b>TOPSIS 근접계수는 {res['TOPSIS_CC']:.3f}</b>로, "
        f"이상적 해에 근접하여 전체 {res['전체_사고유형수']}개 사고유형 중 "
        f"<b>{res['예방우선순위']}위</b>의 예방 우선순위로 산정되었습니다. "
        f"특히 <b>{key_driver}</b>이(가) 우선순위를 끌어올린 핵심 요인입니다."
    )


st.markdown(
    f'<div style="background:#f5f9ff;border:1px solid #cdddf5;border-left:4px solid #0f9d72;'
    f'border-radius:0 12px 12px 0;padding:16px 18px;font-size:15px;'
    f'line-height:1.75;color:#172033;">{priority_reason(r)}</div>',
    unsafe_allow_html=True,
)

st.divider()

# ===========================================================================
# 5. AI Safety Advisor
# ===========================================================================
section_title(4, "AI 예방 자문 (AI Safety Advisor)")
st.caption("학교급·사고장소·사고형태·위험등급·보상금·빈도·우선순위를 종합해 "
           "시설 점검·운영·안전교육을 아우르는 예방 안내를 한 문단으로 알려드립니다.")

if st.button("AI 예방 대책 생성", use_container_width=True):
    ctx = {**ctx_in,
           "위험등급": r["위험등급"], "위험등급_index": r["위험등급_index"],
           "예상보상금": r["예상보상금"], "사고빈도": r["사고빈도"],
           "예방우선순위": r["예방우선순위"], "전체_사고유형수": r["전체_사고유형수"]}
    with st.spinner("AI가 예방 방안을 작성하고 있습니다..."):
        advice, model_label = generate_safety_advice(ctx)
    st.session_state["advice"] = advice
    st.session_state["advice_model"] = model_label

if "advice" in st.session_state:
    advice = st.session_state["advice"]
    model_label = st.session_state.get("advice_model", "규칙 기반(오프라인 폴백)")
    st.markdown(
        f"""
        <div style="background:#f5f9ff;border:1px solid #cdddf5;
                    border-left:4px solid #2563eb;border-radius:0 12px 12px 0;
                    padding:18px 20px;font-size:15.5px;line-height:1.8;color:#172033;">
          {advice}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"본 안내는 {model_label} 모델이 생성하였습니다.")

st.divider()

# ===========================================================================
# 6. Top10 예방 우선관리 사고 (표)
# ===========================================================================
section_title(5, "예방 우선관리 대상 상위 10건")
top10 = get_top10_priority()
show = top10.copy()
show["위험확률"] = (show["위험확률"] * 100).round(1).astype(str) + "%"
show["예상보상금"] = show["예상보상금"].map(lambda v: f"{v:,}원")
show["사고빈도"] = show["사고빈도"].map(lambda v: f"{v:,}건")
show["TOPSIS_CC"] = show["TOPSIS_CC"].round(3)
show = show[["순위", "사고유형", "위험등급", "위험확률", "예상보상금", "사고빈도", "TOPSIS_CC"]]

st.dataframe(
    show, use_container_width=True, hide_index=True,
    column_config={
        "순위": st.column_config.NumberColumn(width="small"),
        "TOPSIS_CC": st.column_config.ProgressColumn(
            "TOPSIS_CC", min_value=0.0, max_value=float(top10["TOPSIS_CC"].max()),
            format="%.3f"),
    },
)
st.caption("TOPSIS 근접계수(CRITIC 가중) 기준 상위 10개 사고유형입니다.")
