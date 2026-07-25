"""
app.py
학교안전사고 예방 의사결정 지원 시스템 (Streamlit)

실행:  streamlit run app.py
"""

import os
import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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

# --- 테마(참고 HTML 디자인) ---------------------------------------------------
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
      font-weight:800; color:var(--navy); margin:2px 0 8px;
    }
    .section-title .badge{
      width:28px; height:28px; border-radius:9px; display:grid; place-items:center;
      background:var(--sky); color:var(--blue); font-size:14px; font-weight:800; flex:none;
    }
    h2, .stMarkdown h2{ color:var(--navy); }

    /* 소주제 흰색 라운드 카드 (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"]{
      background:#fff; border:1px solid var(--line); border-radius:18px;
      box-shadow:0 10px 24px rgba(16,36,62,.06);
      padding:18px 22px 20px; margin-bottom:16px;
    }

    /* 설명칸 — 옅은 남색 라운드 박스 */
    .note{
      background:var(--sky); border:1px solid #cfe0fb; border-radius:12px;
      padding:11px 15px; font-size:13.5px; line-height:1.65; color:#3f5678;
      margin:2px 0 14px;
    }
    .note-strong{
      background:#f5f9ff; border:1px solid #cdddf5; border-left:4px solid var(--blue);
      border-radius:0 12px 12px 0; padding:15px 18px; font-size:15px;
      line-height:1.75; color:var(--text); margin:6px 0 4px;
    }

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

    /* progress bar 색 (안전관리 점검률) */
    div[data-testid="stProgress"] div[role="progressbar"] > div{
      background:linear-gradient(90deg,#4f8df7,#1e5aa0);
    }

    hr{ margin:1.1rem 0; border:none; border-top:1px solid var(--line); }
    [data-testid="stDataFrame"], .stPlotlyChart{ overflow-x:auto; }

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

# 위험등급 색상 & 그라데이션 (등급색 상자, 살짝 그라데이션)
GRADE_COLOR = {
    "1등급_초고위험": "#dc2626", "2등급_심각위험": "#e5484d",
    "3등급_고위험": "#d97706", "4등급_중점관리": "#ca8a04",
    "5등급_일반위험": "#16a34a", "6등급_저위험": "#0f9d72",
}
GRADE_GRADIENT = {
    "1등급_초고위험": ("#ef4444", "#b91c1c"),  # 빨강
    "2등급_심각위험": ("#f36f47", "#c2410c"),  # 주황빨강
    "3등급_고위험":   ("#f59e0b", "#c2620a"),  # 주황
    "4등급_중점관리": ("#eab308", "#b45309"),  # 노랑·주황
    "5등급_일반위험": ("#34d399", "#15803d"),  # 초록
    "6등급_저위험":   ("#10b981", "#0b7c5a"),  # 청록
}


def section_title(num, text):
    st.markdown(
        f'<div class="section-title"><span class="badge">{num}</span>{text}</div>',
        unsafe_allow_html=True,
    )


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


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
    school = st.selectbox("학교급", SCHOOL_LEVELS, index=None,
                          placeholder="학교급을 선택하세요")
    place = st.selectbox("사고장소", PLACES, index=None,
                         placeholder="사고장소를 선택하세요")
    atype = st.selectbox("사고형태", ACCIDENT_TYPES, index=None,
                         placeholder="사고형태를 선택하세요")
    run = st.button("예방 우선순위 분석", type="primary", use_container_width=True)
    st.divider()
    st.caption("입력 항목 : 학교급 · 사고장소 · 사고형태")

# 분석 버튼을 눌렀지만 3개를 모두 고르지 않은 경우
if run and not (school and place and atype):
    st.warning("학교급 · 사고장소 · 사고형태를 모두 선택한 뒤 "
               "[예방 우선순위 분석]을 눌러 주십시오.")
    st.stop()

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
grade_grad = GRADE_GRADIENT.get(grade, ("#14345c", "#1e5aa0"))

st.markdown(
    f'<div class="target-tag">분석 대상 &nbsp;|&nbsp; '
    f'{ctx_in["학교급"]} · {ctx_in["사고장소"]} · {ctx_in["사고형태"]}</div>',
    unsafe_allow_html=True,
)

# ===========================================================================
# 1. 결과 카드 4종 (위험등급 = 등급색 그라데이션 카드)
# ===========================================================================
c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1])


def hero_card(col, label, value, sub, c_from, c_to):
    col.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{c_from},{c_to});color:#fff;
                    border:none;border-radius:17px;padding:18px 19px;min-height:120px;
                    margin-bottom:10px;box-shadow:0 10px 24px rgba(16,36,62,.16);
                    text-shadow:0 1px 2px rgba(0,0,0,.22);">
          <div style="font-size:12px;font-weight:750;color:rgba(255,255,255,.86);
                      text-transform:uppercase;letter-spacing:.06em;">{label}</div>
          <div style="font-size:clamp(19px,4.4vw,23px);font-weight:850;
                      margin-top:8px;line-height:1.25;">{value}</div>
          <div style="font-size:12px;color:rgba(255,255,255,.9);
                      margin-top:9px;line-height:1.45;">{sub}</div>
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
          f"6단계 중 {r['위험등급_index'] + 1}단계 · 상대적 개선 우선순위",
          grade_grad[0], grade_grad[1])
metric_card(c2, "예상 보상금", f"{r['예상보상금']:,}원",
            "사고 1건당 예측 총보상액", "#1e5aa0")
metric_card(c3, "사고 빈도", f"{r['사고빈도']:,}건",
            "연간 발생 추정 건수", "#0f9d72")
metric_card(c4, "예방 우선순위", f"{r['예방우선순위']}위",
            f"전체 {r['전체_사고유형수']}개 사고유형 중", "#dc2626")

# ===========================================================================
# 2. 위험도 시각화
# ===========================================================================
norm = r["지표_정규화"]
with st.container(border=True):
    section_title(1, "위험도 시각화")
    note("왼쪽은 이 사고유형의 종합 예방 우선순위 점수(TOPSIS), "
         "오른쪽은 그 점수를 만들어 낸 세부 지표(예방 우선순위 선정 근거)입니다.")

    g1, g2 = st.columns([1, 1.1])

    with g1:
        gfig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(norm["TOPSIS"] * 100, 1),
            number={"suffix": "%", "font": {"color": "#10243e"}},
            title={"text": "TOPSIS 근접계수", "font": {"size": 16, "color": "#10243e"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e5aa0"},
                "bgcolor": "#eaf2ff",
                "borderwidth": 0,
                "steps": [{"range": [0, 100], "color": "#eaf2ff"}],
            },
        ))
        st.plotly_chart(_theme(gfig, height=250), use_container_width=True)
        st.markdown(
            "**TOPSIS 근접계수** : 위험확률·예상보상금·사고빈도 세 지표를 결합하여, "
            "'가장 시급히 관리해야 할 이상적 사고유형'에 얼마나 가까운지를 "
            "**0~100%로 나타낸 종합 우선순위 점수**입니다. 100%에 가까울수록 "
            "예방 우선순위가 높습니다."
        )

    with g2:
        metrics = ["위험확률", "예상보상금", "사고빈도"]
        vals = [norm["위험확률"], norm["예상보상금"], norm["사고빈도"]]
        bar = go.Figure(go.Bar(
            x=[v * 100 for v in vals], y=metrics, orientation="h",
            marker=dict(color="#1e5aa0", cornerradius=9),
            width=0.5,
            text=[f"{v * 100:.0f}%" for v in vals],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=13, color="#172033"),
        ))
        bar.update_layout(
            title="예방 우선순위 선정 근거",
            xaxis=dict(range=[0, 118], showgrid=False, zeroline=False,
                       showticklabels=False, title=None),
            yaxis=dict(autorange="reversed", tickfont=dict(size=14, color="#172033")),
            bargap=0.5,
        )
        st.plotly_chart(_theme(bar, height=270, top=52), use_container_width=True)
        st.caption("위 TOPSIS 점수를 구성하는 세 지표입니다. "
                   "각 막대가 길수록 그 지표가 예방 우선순위를 끌어올린 것입니다.")

# ===========================================================================
# 3. SHAP 분석
# ===========================================================================
shap_df = pd.DataFrame(r["shap"])
top_shap = shap_df.iloc[0]

with st.container(border=True):
    section_title(2, "SHAP 분석 — 무엇이 이 사고의 위험을 높였나")
    note("SHAP은 위험등급 예측에 각 입력 요인(학교급·사고장소·사고형태)이 "
         "<b>얼마나, 어느 방향으로 기여했는지</b>를 수치로 설명합니다. "
         "값이 클수록(+) 위험등급을 더 크게 끌어올린 요인입니다.")

    sh1, sh2 = st.columns([1.6, 1])
    with sh1:
        shap_max = float(shap_df["기여도"].max())
        sfig = go.Figure(go.Bar(
            x=shap_df["기여도"], y=shap_df["요인"], orientation="h",
            marker=dict(color="#1e5aa0", cornerradius=9), width=0.6,
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
            bargap=0.35,
        )
        st.plotly_chart(_theme(sfig, height=380, top=55), use_container_width=True)

    with sh2:
        st.markdown("**요인별 기여도**")
        for _, row in shap_df.iterrows():
            st.metric(row["요인"], f"+{row['기여도']:.3f}", "위험 상승 기여")

    st.markdown(
        f'<div class="note-strong">가장 큰 위험 상승 요인은 <b>{top_shap["요인"]}</b> '
        f'(기여도 +{top_shap["기여도"]:.3f})으로, 위험등급 예측을 높이는 방향으로 '
        f'가장 강하게 작용했습니다. 이 요인을 우선적으로 관리하면 위험을 가장 효과적으로 '
        f'낮출 수 있습니다.</div>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# 4. 예방 우선순위 선정 이유
# ===========================================================================
with st.container(border=True):
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
        f'<div class="note-strong" style="border-left-color:#0f9d72;">'
        f'{priority_reason(r)}</div>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# 5. AI Safety Advisor
# ===========================================================================
with st.container(border=True):
    section_title(4, "AI 예방 자문 (AI Safety Advisor)")
    note("학교급·사고장소·사고형태·위험등급·보상금·빈도·우선순위를 종합해 "
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
            f'<div class="note-strong">{advice}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"본 안내는 {model_label} 모델이 생성하였습니다.")

# ===========================================================================
# 6. 현장 점검 체크리스트
# ===========================================================================
with st.container(border=True):
    section_title(5, "현장 점검 체크리스트")
    note("아래 항목을 직접 점검·체크하면 안전관리 점검률이 자동으로 계산됩니다. "
         "미점검 항목은 하단에 정리되어 표시됩니다.")

    _s, _p, _t = ctx_in["학교급"], ctx_in["사고장소"], ctx_in["사고형태"]
    checklist_groups = [
        ("1. 시설·환경 점검", "", [
            "사고 발생 장소(운동장, 강당, 계단 등)의 시설물 파손 여부를 확인하였는가?",
            "바닥의 미끄럼, 요철, 장애물 등 위험요인을 제거하였는가?",
            "안전표지 및 위험구역 표시가 적절히 설치되어 있는가?",
            "조명 및 시야 확보 상태가 양호한가?",
            "보호시설(매트, 난간 등)이 정상적으로 설치되어 있는가?",
        ]),
        ("2. 교육·관리 점검", "", [
            "사고 유형에 맞는 안전교육을 실시하였는가?",
            "학생들에게 안전수칙을 사전에 안내하였는가?",
            "담당 교사의 안전관리 계획이 수립되어 있는가?",
            "사고 발생 가능성이 높은 활동에 대해 충분한 감독 인력이 배치되어 있는가?",
        ]),
        ("3. 법률 기반 점검", "학교안전사고 예방 및 보상에 관한 법률 반영", [
            "학교안전사고 예방계획에 해당 활동이 포함되어 있는가?",
            "사고 위험요인에 대한 사전 점검을 실시하였는가?",
            "필요한 경우 관계기관과 협조체계를 구축하였는가?",
            "학교 밖 교육활동인 경우 사전 현장답사를 실시하였는가?",
        ]),
    ]

    checked_flags, unchecked, _gi = [], [], 0
    for _gtitle, _gsub, _items in checklist_groups:
        _sub = (f' <span style="font-weight:500;color:#8894a6;font-size:12px;">'
                f'({_gsub})</span>') if _gsub else ""
        st.markdown(
            f'<div style="font-weight:800;color:#10243e;font-size:15px;'
            f'margin:12px 0 2px;">{_gtitle}{_sub}</div>',
            unsafe_allow_html=True,
        )
        for _item in _items:
            _c = st.checkbox(_item, key=f"chk_{_s}_{_p}_{_t}_{_gi}")
            checked_flags.append(_c)
            if not _c:
                unchecked.append(_item)
            _gi += 1

    done = sum(checked_flags)
    total = len(checked_flags)
    rate = (done / total * 100) if total else 0.0

    st.progress(done / total if total else 0.0,
                text=f"안전관리 점검률  {rate:.0f}%   ({done}/{total} 완료)")
    if unchecked:
        items_html = "".join(
            f'<div style="margin:6px 0;">• {it}</div>' for it in unchecked
        )
        st.markdown(
            f'<div class="note-strong" style="border-left-color:#d97706;'
            f'background:#fff8ed;border-color:#f3dcb4;">'
            f'<b>미점검 항목</b><div style="margin-top:8px;font-size:14px;'
            f'color:#5b4a2a;">{items_html}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="note-strong" style="border-left-color:#0f9d72;'
            'background:#effaf4;border-color:#bfe8d4;color:#0b5c40;">'
            '<b>모든 안전 점검이 완료되었습니다.</b><br>'
            '현재 안전관리 수준이 양호합니다.</div>',
            unsafe_allow_html=True,
        )

# ===========================================================================
# 7. Top10 예방 우선관리 사고 (표)
# ===========================================================================
with st.container(border=True):
    section_title(6, "예방 우선관리 대상 상위 10건")
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

# ===========================================================================
# 8. 관련 안전 지침 (PDF 다운로드·미리보기)
# ===========================================================================
# ▼▼ 여기 두 개의 "drive"(또는 url / path)만 채우면 됩니다 ▼▼
#  대용량(수백 MB) PDF는 GitHub/Streamlit Cloud 레포에 직접 올릴 수 없습니다(파일당 100MB 제한).
#  → 구글 드라이브에 올린 뒤 '링크가 있는 모든 사용자'로 공유하고,
#     아래 drive 에 '공유 링크' 또는 '파일 ID'를 붙여넣으세요.
#     공유 링크 예:  https://drive.google.com/file/d/1AbCdEfGhIjK.../view?usp=sharing
#     파일 ID  예:  1AbCdEfGhIjK...
#  소용량 PDF라면 url(직접 PDF 링크) 또는 path(레포 내 파일 경로)를 대신 넣어도 됩니다.
GUIDES = [
    
    {"title": "학교안전사고 예방 및 보상에 관한 법률",
     "drive": "https://drive.google.com/file/d/1lFauN72iIcap3ZQKh3cJe93mCr3LkWdA/view?usp=sharing", "url": "", "path": ""},
    
    {"title": "학교안전교육 7대 표준안 교육 자료집",
     "items": [
         {"title": "유치원", "drive": "https://drive.google.com/file/d/1vvWp8s7wNWqsHuUKXVg_QIdZMVrrxxrj/view?usp=sharing", "url": "", "path": ""},
         {"title": "초등 1-2학년", "drive": "https://drive.google.com/file/d/19-HVbU7cfZ9jrYIGsXvZb4IxzX9DTfrS/view?usp=sharing","url": "", "path": ""},
         {"title": "초등 3-4학년", "drive":"https://drive.google.com/file/d/1f2vhrS3GPFHnQVFnDt3oDXRES9JDc0e0/view?usp=sharing","url": "", "path": ""},
         {"title": "초등 5-6학년", "drive": "https://drive.google.com/file/d/18gHnPZYP5ZGewE3GidJ2jeqq2_LCZARM/view?usp=sharing","url": "", "path": ""},
         {"title": "중등", "drive": "https://drive.google.com/file/d/1IG4VrT2SpoE-i0FTvxts-7Ralc2ntnOS/view?usp=sharing","url": "", "path": ""},
    ]},
]


def _drive_id(s: str) -> str:
    """구글 드라이브 공유 링크 또는 파일 ID 문자열에서 파일 ID만 추출."""
    import re
    s = (s or "").strip()
    if not s:
        return ""
    if "drive.google.com" in s:
        m = re.search(r"/d/([A-Za-z0-9_-]+)", s) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", s)
        return m.group(1) if m else ""
    return s  # 이미 파일 ID 형태


def _pdf_iframe(src: str):
    components.html(
        f'<iframe src="{src}" width="100%" height="640" '
        f'style="border:1px solid #dbe3ef;border-radius:12px;"></iframe>',
        height=660,
    )


def guide_block(g: dict):
    st.markdown(
        f'<div style="font-weight:800;color:#10243e;font-size:15px;'
        f'margin:12px 0 6px;">📘 {g["title"]}</div>',
        unsafe_allow_html=True,
    )
    did = _drive_id(g.get("drive", ""))
    url = (g.get("url") or "").strip()
    path = (g.get("path") or "").strip()
    key = g["title"]
    ca, cb = st.columns(2)

    # 다운로드 버튼
    with ca:
        if did:
            st.link_button(
                "📥 다운로드",
                f"https://drive.google.com/uc?export=download&id={did}",
                use_container_width=True,
            )
        elif url:
            st.link_button("📥 다운로드", url, use_container_width=True)
        elif path and os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button(
                    "📥 다운로드", f, file_name=os.path.basename(path),
                    mime="application/pdf", use_container_width=True,
                    key=f"dl_{key}",
                )
        else:
            st.button("📥 다운로드 (경로 미설정)", disabled=True,
                      use_container_width=True, key=f"dl_{key}")

    # 미리보기 버튼(토글)
    with cb:
        show = st.toggle("👁 미리보기", key=f"pv_{key}")

    if show:
        if did:
            _pdf_iframe(f"https://drive.google.com/file/d/{did}/preview")
        elif url:
            _pdf_iframe(url)
        elif path and os.path.exists(path):
            if os.path.getsize(path) < 20_000_000:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                _pdf_iframe(f"data:application/pdf;base64,{b64}")
            else:
                st.info("20MB 이상 로컬 PDF는 미리보기가 제한됩니다. "
                        "구글 드라이브(drive) 방식을 사용해 주세요.")
        else:
            st.info("소스가 설정되지 않았습니다. GUIDES 의 drive/url/path "
                    "중 하나를 채워 주세요.")


with st.container(border=True):
    section_title(7, "관련 안전 지침")
    note("아래 문서를 클릭해 <b>다운로드</b>하거나 <b>미리보기</b>할 수 있습니다. "
         "대용량 PDF는 구글 드라이브 링크를 연결해 주세요.")
    for _g in GUIDES:
        guide_block(_g)
        st.write("")
