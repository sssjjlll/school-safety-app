"""
backend.py
학교안전사고 예방 의사결정 지원 시스템의 분석 백엔드.

노트북(공모전_셀_통합본)에서 학습한 3개 모델의 결과를 하나의 함수로 묶는다.
  - 위험등급 분류 모델(LightGBM)           -> 위험등급 / 위험확률(X1)
  - 예상 보상금 회귀 모델(LightGBM/CatBoost) -> 예상 보상금(X2)
  - 사고유형별 집계                          -> 사고 빈도(X3)
  - CRITIC-TOPSIS                            -> 예방 우선순위(TOPSIS_CC)
  - SHAP(TreeExplainer)                      -> 위험도 상승 기여 요인

★ 실제 배포 시:
  analyze_accident() 내부의 `_MOCK` 블록을 실제 모델 추론 코드로 교체하면 된다.
  반환 딕셔너리의 "키 계약(contract)"만 지키면 app.py는 수정할 필요가 없다.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 입력 카테고리 (노트북 FEATURE_COLS 기준)
# ---------------------------------------------------------------------------
SCHOOL_LEVELS = ["초등학교", "중학교", "고등학교"]
PLACES = ["운동장", "강당", "체육관", "계단", "복도", "교실", "스쿨존"]
ACCIDENT_TYPES = ["넘어짐", "부딪힘", "떨어짐", "물체", "교통"]

# 위험등급 라벨 (노트북 6단계 타깃, index 0 = 가장 위험)
RISK_GRADES = [
    "1등급_초고위험",
    "2등급_심각위험",
    "3등급_고위험",
    "4등급_중점관리",
    "5등급_일반위험",
    "6등급_저위험",
]

# SHAP 예방전략 규칙 (노트북 strategy_rules 원본)
STRATEGY_RULES = {
    "사고장소": {
        "운동장": "활동 구역 분리, 충돌 위험 활동 관리, 쉬는시간 순찰 강화",
        "강당": "바닥 미끄럼과 시설물 점검, 종목별 위험구역 표시",
        "체육관": "바닥 상태와 보호장비 점검, 활동 전 안전수칙 안내",
        "계단": "미끄럼 방지 시설 점검, 우측통행 지도, 혼잡시간 관리",
        "복도": "뛰기 방지 지도, 이동 동선 분리, 쉬는시간 순찰 강화",
        "교실": "책상·의자 배치와 모서리 점검, 장난 및 충돌 예방교육",
        "스쿨존": "차량·보행 동선 분리, 등하교 교통안전지도 강화",
    },
    "사고형태": {
        "넘어짐": "미끄럼·장애물 점검, 이동수칙 교육, 준비운동 강화",
        "부딪힘": "활동 공간 확보, 충돌 구역 분리, 과속 이동과 장난 예방",
        "떨어짐": "난간·계단·놀이시설 점검, 높이 활동 감독 강화",
        "물체": "기구 고정과 보관상태 점검, 보호장비 착용 지도",
        "교통": "스쿨존 관리, 차량·보행 동선 분리, 교통안전교육 강화",
    },
    "학교급": {
        "초등학교": "발달단계를 고려한 반복 안전교육과 생활지도 강화",
        "중학교": "체육·쉬는시간의 활동성 높은 사고 예방지도 강화",
        "고등학교": "종목별 체육활동 안전수칙과 시설 점검 강화",
    },
}

# 사고유형 조합 총 개수 (학교급 3 × 장소 7 × 형태 5)
TOTAL_TYPES = len(SCHOOL_LEVELS) * len(PLACES) * len(ACCIDENT_TYPES)


# ---------------------------------------------------------------------------
# 내부 유틸: 입력 조합을 재현 가능한 난수 시드로 변환 (mock 전용)
# ---------------------------------------------------------------------------
def _seed(*parts: str) -> int:
    key = "|".join(parts)
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def _mock_analyze(school: str, place: str, acc_type: str) -> dict:
    """
    실제 모델이 없을 때 사용하는 재현 가능한 가상 추론.
    동일 입력 -> 동일 결과. 도메인 상식(운동장·넘어짐이 고위험)을 반영한다.
    """
    rng = np.random.default_rng(_seed(school, place, acc_type))

    # 도메인 가중치: 노트북 EDA 결론을 반영 (운동장/강당/체육관 + 넘어짐/부딪힘이 위험)
    place_w = {"운동장": 0.9, "강당": 0.8, "체육관": 0.78, "계단": 0.6,
               "복도": 0.45, "교실": 0.4, "스쿨존": 0.55}
    type_w = {"넘어짐": 0.85, "부딪힘": 0.7, "떨어짐": 0.75, "물체": 0.5, "교통": 0.65}
    level_w = {"중학교": 0.85, "고등학교": 0.75, "초등학교": 0.6}

    base = 0.4 * place_w[place] + 0.4 * type_w[acc_type] + 0.2 * level_w[school]
    noise = rng.normal(0, 0.05)

    risk_prob = float(np.clip(base + noise, 0.02, 0.98))              # X1 위험확률
    # 예상 보상금: 위험확률 + 고액사고(떨어짐/교통) 가중, 로그정규 분포
    sev_boost = {"떨어짐": 1.8, "교통": 2.2, "부딪힘": 1.2,
                 "넘어짐": 1.0, "물체": 1.1}[acc_type]
    comp = float(np.expm1(np.log1p(120_000) + risk_prob * 3.2) * sev_boost)  # X2 원
    comp = round(comp, -3)
    # 사고 빈도: 넘어짐/운동장이 압도적 (노트북 결론)
    freq = int(np.clip(rng.normal(base * 1500, 120), 20, 2000))       # X3 건

    # 위험등급: 위험확률 분위로 매핑
    if risk_prob >= 0.85:
        grade_idx = 0
    elif risk_prob >= 0.72:
        grade_idx = 1
    elif risk_prob >= 0.58:
        grade_idx = 2
    elif risk_prob >= 0.45:
        grade_idx = 3
    elif risk_prob >= 0.30:
        grade_idx = 4
    else:
        grade_idx = 5

    # 정규화 지표(전체 사고유형 대비 상대 위치, 0~1) — 게이지/비교용
    norm = {
        "위험확률": risk_prob,
        "예상보상금": float(np.clip(np.log1p(comp) / np.log1p(50_000_000), 0, 1)),
        "사고빈도": float(np.clip(freq / 2000, 0, 1)),
    }
    # TOPSIS_CC: 세 지표의 CRITIC 가중 근접도 (근사)
    topsis_cc = float(np.clip(
        0.45 * norm["위험확률"] + 0.30 * norm["예상보상금"] + 0.25 * norm["사고빈도"]
        + rng.normal(0, 0.02), 0.02, 0.98))
    norm["TOPSIS"] = topsis_cc

    # 예방 우선순위(전체 조합 중 순위): TOPSIS_CC 높을수록 상위
    rank = int(round((1 - topsis_cc) * (TOTAL_TYPES - 1))) + 1

    # SHAP 기여 요인 (위험등급 상승 방향 기여도)
    shap = [
        {"요인": f"사고장소 = {place}", "기여도": round(place_w[place] * 0.18 + rng.normal(0, 0.01), 4)},
        {"요인": f"사고형태 = {acc_type}", "기여도": round(type_w[acc_type] * 0.16 + rng.normal(0, 0.01), 4)},
        {"요인": f"학교급 = {school}", "기여도": round(level_w[school] * 0.11 + rng.normal(0, 0.01), 4)},
    ]
    shap.sort(key=lambda d: d["기여도"], reverse=True)

    return {
        "위험등급": RISK_GRADES[grade_idx],
        "위험등급_index": grade_idx,
        "위험확률": risk_prob,
        "예상보상금": int(comp),
        "사고빈도": freq,
        "TOPSIS_CC": topsis_cc,
        "예방우선순위": rank,
        "전체_사고유형수": TOTAL_TYPES,
        "지표_정규화": norm,
        "shap": shap,
    }


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def analyze_accident(school_level: str, place: str, accident_type: str) -> dict:
    """
    학교급·사고장소·사고형태를 입력받아 예방 의사결정 지표를 반환한다.

    Returns (키 계약):
        위험등급        : str   ex) "1등급_초고위험"
        위험등급_index  : int   0=가장 위험 ~ 5=가장 안전
        위험확률        : float 0~1  (X1_High_Risk_Prob)
        예상보상금      : int   원   (X2_Predicted_Compensation)
        사고빈도        : int   건   (X3_Accident_Frequency)
        TOPSIS_CC       : float 0~1  (예방 우선순위 근접계수)
        예방우선순위    : int   전체 사고유형 중 순위 (1=최우선)
        전체_사고유형수 : int
        지표_정규화     : dict  {위험확률, 예상보상금, 사고빈도, TOPSIS} 각 0~1
        shap            : list[{"요인": str, "기여도": float}]
    """
    # === 실제 모델 연결 지점 ===================================================
    # 예)
    #   x = build_feature_row(school_level, place, accident_type)
    #   grade_idx = risk_clf.predict(x)[0]
    #   risk_prob = risk_clf.predict_proba(x)[0, TARGET_CLASS_INDEX]
    #   comp = np.expm1(comp_reg.predict(x)[0])
    #   ...
    # ==========================================================================
    return _mock_analyze(school_level, place, accident_type)


# ---------------------------------------------------------------------------
# Top10 예방 우선관리 사고 (노트북 prevention_priority_new_X2 대응)
# ---------------------------------------------------------------------------
def get_top10_priority() -> pd.DataFrame:
    """전체 사고유형 조합을 분석해 TOPSIS_CC 상위 10개를 반환한다."""
    rows = []
    for s in SCHOOL_LEVELS:
        for p in PLACES:
            for t in ACCIDENT_TYPES:
                r = analyze_accident(s, p, t)
                rows.append({
                    "사고유형": f"{s}-{p}-{t}",
                    "학교급": s,
                    "사고장소": p,
                    "사고형태": t,
                    "위험등급": r["위험등급"],
                    "위험확률": r["위험확률"],
                    "예상보상금": r["예상보상금"],
                    "사고빈도": r["사고빈도"],
                    "TOPSIS_CC": r["TOPSIS_CC"],
                })
    df = pd.DataFrame(rows).sort_values("TOPSIS_CC", ascending=False).reset_index(drop=True)
    df.insert(0, "순위", df.index + 1)
    return df.head(10)


# ---------------------------------------------------------------------------
# 예방전략 초안 (SHAP strategy_rules 기반) — AI Advisor 폴백에 사용
# ---------------------------------------------------------------------------
def recommend_strategy(feature: str, category: str) -> str:
    default = "현장 점검과 맞춤형 안전교육을 강화"
    for keyword, strategy in STRATEGY_RULES.get(feature, {}).items():
        if keyword in str(category):
            return strategy
    return default
