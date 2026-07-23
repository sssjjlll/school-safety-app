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

# --- 모바일 반응형 스타일 -----------------------------------------------------
st.markdown(
    """
    <style>
    /* 좁은 화면에서 본문 여백 축소 */
    @media (max-width: 640px) {
        .block-container { padding: 1rem 0.8rem 3rem !important; }
        h1 { font-size: 1.5rem !important; }
        h3 { font-size: 1.15rem !important; }
        /* st.columns 를 세로로 쌓아 카드가 찌그러지지 않게 */
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
    }
    /* 표·그래프가 화면을 넘으면 가로 스크롤 */
    [data-testid="stDataFrame"], .stPlotlyChart { overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

GRADE_COLOR = {
    "1등급_초고위험": "#c0392b",
    "2등급_심각위험": "#e74c3c",
    "3등급_고위험": "#e67e22",
    "4등급_중점관리": "#f1c40f",
    "5등급_일반위험": "#2ecc71",
    "6등급_저위험": "#27ae60",
}

st.title("🏫 학교안전사고 예방 의사결정 지원 시스템")
st.caption("위험등급 분류 · 예상 보상금 회귀 · CRITIC-TOPSIS 우선순위 · SHAP 해석 기반")

# ---------------------------------------------------------------------------
# 입력 영역
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 사고 조건 입력")
    school = st.selectbox("학교급", SCHOOL_LEVELS, index=1)
    place = st.selectbox("사고장소", PLACES, index=0)
    atype = st.selectbox("사고형태", ACCIDENT_TYPES, index=0)
    run = st.button("📊 예방 우선순위 분석", type="primary", use_container_width=True)
    st.divider()
    st.caption("입력: 학교급 · 사고장소 · 사고형태")

if not run and "result" not in st.session_state:
    st.info("좌측에서 **학교급 · 사고장소 · 사고형태**를 선택하고 "
            "**[예방 우선순위 분석]** 버튼을 눌러주세요.")
    st.stop()

if run:
    st.session_state["result"] = analyze_accident(school, place, atype)
    st.session_state["ctx"] = {"학교급": school, "사고장소": place, "사고형태": atype}
    st.session_state.pop("advice", None)  # 새 분석 시 조언 초기화

r = st.session_state["result"]
ctx_in = st.session_state["ctx"]
grade = r["위험등급"]
grade_color = GRADE_COLOR.get(grade, "#7f8c8d")

st.subheader(f"분석 대상 : {ctx_in['학교급']} · {ctx_in['사고장소']} · {ctx_in['사고형태']}")

# ===========================================================================
# 1. 결과 카드 4종
# ===========================================================================
c1, c2, c3, c4 = st.columns(4)


def card(col, label, value, sub, color):
    col.markdown(
        f"""
        <div style="background:{color}15;border:1px solid {color}55;
                    border-radius:14px;padding:16px;min-height:120px;
                    margin-bottom:10px;">
          <div style="font-size:13px;color:#666;">{label}</div>
          <div style="font-size:clamp(20px,5vw,26px);font-weight:800;color:{color};
                      margin-top:6px;line-height:1.2;">{value}</div>
          <div style="font-size:12px;color:#888;margin-top:6px;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


card(c1, "위험등급", grade.replace("_", " "),
     f"6단계 중 {r['위험등급_index'] + 1}단계", grade_color)
card(c2, "예상 보상금", f"{r['예상보상금']:,}원",
     "사고 1건당 예측 총보상액", "#8e44ad")
card(c3, "사고 빈도", f"{r['사고빈도']:,}건",
     "연간 발생 추정 건수", "#2980b9")
card(c4, "예방 우선순위", f"{r['예방우선순위']}위",
     f"전체 {r['전체_사고유형수']}개 사고유형 중", "#16a085")

st.divider()

# ===========================================================================
# 2. 위험도 시각화 (위험확률 · TOPSIS · 예상보상금 · 사고빈도 · SHAP)
# ===========================================================================
st.markdown("### 📈 위험도 시각화")
st.caption(
    "이 사고유형의 위험 수준을 4가지 지표로 나눠 보여줍니다. "
    "**게이지 2개**는 값 자체(절대 수준)를, **오른쪽 막대**는 전체 사고유형과 "
    "비교한 상대 위치(0~100)를 나타냅니다."
)
norm = r["지표_정규화"]

with st.expander("ℹ️ 각 지표가 무엇을 의미하나요?"):
    st.markdown(
        "- **위험 확률 (X1)** : 이 사고유형이 고위험 등급으로 분류될 예측 확률입니다. "
        "높을수록 심각한 사고로 이어질 가능성이 큽니다.\n"
        "- **TOPSIS 근접계수** : 위험확률·예상보상금·사고빈도 세 지표를 CRITIC 가중치로 "
        "결합해, '가장 시급히 관리해야 할 이상적 사고유형'에 얼마나 가까운지를 0~1로 나타낸 "
        "종합 우선순위 점수입니다. 1에 가까울수록 예방 우선순위가 높습니다.\n"
        "- **예상 보상금 (X2)** : 사고 1건당 예측되는 총보상액입니다. 사고의 *심각도(피해 규모)*를 대변합니다.\n"
        "- **사고 빈도 (X3)** : 해당 사고유형의 연간 발생 추정 건수로, 사고의 *발생 빈도*를 대변합니다.\n\n"
        "막대의 **상대 위치**는 전체 사고유형 중 이 사고가 어느 위치인지를 보여줍니다. "
        "예: 위험확률 막대가 90이면 전체 상위 10% 수준의 고위험이라는 의미입니다."
    )

g1, g2 = st.columns([1.1, 1])

with g1:
    # 위험확률 & TOPSIS 게이지
    gg1, gg2 = st.columns(2)

    def gauge(title, value, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            number={"suffix": "%"},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#eafaf1"},
                    {"range": [40, 70], "color": "#fef9e7"},
                    {"range": [70, 100], "color": "#fdedec"},
                ],
            },
        ))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=45, b=10))
        return fig

    gg1.plotly_chart(gauge("위험 확률 (X1)", r["위험확률"], "#e74c3c"),
                     use_container_width=True)
    gg2.plotly_chart(gauge("TOPSIS 근접계수", norm["TOPSIS"], "#16a085"),
                     use_container_width=True)
    st.caption("🔴 **위험 확률**: 고위험 분류 확률(높을수록 위험)  ·  "
               "🟢 **TOPSIS**: 세 지표를 결합한 종합 예방 우선순위 점수")

with g2:
    # 예상보상금 · 사고빈도 상대 위치 막대
    bar = go.Figure()
    metrics = ["예상보상금", "사고빈도", "위험확률", "TOPSIS"]
    vals = [norm["예상보상금"], norm["사고빈도"], norm["위험확률"], norm["TOPSIS"]]
    colors = ["#8e44ad", "#2980b9", "#e74c3c", "#16a085"]
    bar.add_trace(go.Bar(
        x=[v * 100 for v in vals], y=metrics, orientation="h",
        marker_color=colors,
        text=[f"{v*100:.0f}" for v in vals], textposition="outside",
    ))
    bar.update_layout(
        title="전체 사고유형 대비 상대 지표 (0~100)",
        xaxis=dict(range=[0, 115], title="상대 위치 (0=최저, 100=최고)"),
        height=290, margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(bar, use_container_width=True)
    st.caption("네 지표를 전체 사고유형과 비교한 **상대 위치**입니다. "
               "값이 클수록(오른쪽) 전체에서 위험·시급한 상위권임을 뜻합니다.")

st.divider()

# ===========================================================================
# 3. SHAP 분석 (기여 요인 시각화 + 중요 요인 요약 통합)
# ===========================================================================
st.markdown("### 🔬 SHAP 분석 — 무엇이 이 사고의 위험을 높였나")
st.caption(
    "SHAP은 위험등급 예측에 각 입력 요인(학교급·사고장소·사고형태)이 "
    "**얼마나, 어느 방향으로 기여했는지**를 수치로 설명합니다. "
    "값이 클수록(+) 위험등급을 더 크게 끌어올린 요인입니다."
)

shap_df = pd.DataFrame(r["shap"])
top_shap = shap_df.iloc[0]

sh1, sh2 = st.columns([1.3, 1])

with sh1:
    sfig = go.Figure(go.Bar(
        x=shap_df["기여도"], y=shap_df["요인"], orientation="h",
        marker_color="#c0392b",
        text=[f"+{v:.3f}" for v in shap_df["기여도"]], textposition="outside",
    ))
    sfig.update_layout(
        title="요인별 위험등급 상승 기여도 (SHAP value)",
        xaxis_title="위험등급 상승 기여도 (클수록 위험 ↑)",
        height=260, margin=dict(l=10, r=10, t=45, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(sfig, use_container_width=True)

with sh2:
    st.markdown("**요인별 기여도**")
    for _, row in shap_df.iterrows():
        st.metric(row["요인"], f"+{row['기여도']:.3f}", "위험 상승 기여")

st.info(
    f"가장 큰 위험 상승 요인은 **{top_shap['요인']}** "
    f"(기여도 +{top_shap['기여도']:.3f})으로, 위험등급 예측을 높이는 방향으로 "
    f"가장 강하게 작용했습니다. 이 요인을 우선적으로 관리하면 위험을 가장 효과적으로 낮출 수 있습니다."
)

st.divider()

# ===========================================================================
# 4. 예방 우선순위 선정 이유 (문장)
# ===========================================================================
st.markdown("### 🧭 예방 우선순위 선정 이유")


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
        f"이 사고유형은 위험 확률이 **{res['위험확률']*100:.0f}%({lvl(n['위험확률'])})**, "
        f"사고 빈도가 **{res['사고빈도']:,}건({lvl(n['사고빈도'])})**, "
        f"예상 보상금이 **{res['예상보상금']:,}원({lvl(n['예상보상금'])})** 수준입니다. "
        f"CRITIC 가중치로 세 지표를 결합한 **TOPSIS 근접계수는 {res['TOPSIS_CC']:.3f}**로, "
        f"이상적 해에 근접하여 전체 {res['전체_사고유형수']}개 사고유형 중 "
        f"**{res['예방우선순위']}위**의 예방 우선순위로 산정되었습니다. "
        f"특히 **{key_driver}**이(가) 우선순위를 끌어올린 핵심 요인입니다."
    )


st.success(priority_reason(r))

st.divider()

# ===========================================================================
# 5. AI Safety Advisor
# ===========================================================================
st.markdown("### 🤖 AI Safety Advisor")
st.caption("학교급·사고장소·사고형태·위험등급·보상금·빈도·우선순위를 종합해 "
           "시설 점검·운영·안전교육을 아우르는 예방 안내를 한 문단으로 알려드립니다.")

if st.button("🧠 AI 예방 대책 생성", use_container_width=True):
    ctx = {**ctx_in,
           "위험등급": r["위험등급"], "위험등급_index": r["위험등급_index"],
           "예상보상금": r["예상보상금"], "사고빈도": r["사고빈도"],
           "예방우선순위": r["예방우선순위"], "전체_사고유형수": r["전체_사고유형수"]}
    with st.spinner("AI가 예방 방법을 정리하고 있어요..."):
        advice, model_label = generate_safety_advice(ctx)
    st.session_state["advice"] = advice
    st.session_state["advice_model"] = model_label

if "advice" in st.session_state:
    advice = st.session_state["advice"]
    model_label = st.session_state.get("advice_model", "규칙 기반(오프라인 폴백)")
    st.markdown(
        f"""
        <div style="background:#eef6ff;border:1px solid #cfe3fb;border-radius:14px;
                    padding:18px 20px;font-size:16px;line-height:1.75;color:#1f2d3d;">
          🧑‍🏫 {advice}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"🤖 이 안내는 **{model_label}** 모델이 작성했습니다.")

st.divider()

# ===========================================================================
# 6. Top10 예방 우선관리 사고 (표)
# ===========================================================================
st.markdown("### 🏆 Top10 예방 우선관리 사고")
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
