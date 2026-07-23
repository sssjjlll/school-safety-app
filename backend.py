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
from functools import lru_cache

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 입력 카테고리 (실제 학교안전사고 데이터 accident.xlsx 전체 고유값, 빈도순)
# ---------------------------------------------------------------------------
SCHOOL_LEVELS = ["초등학교", "중학교", "고등학교", "유치원", "특수학교", "기타학교"]

PLACES = [
    "강당(체육관)", "운동장", "일반(교과)교실", "계단", "복도",
    "기타 체육·집회공간", "놀이터", "특별교실(과학실 외)", "실·내외 체육시설",
    "기타 교내", "화장실", "기타 공용공간", "급식실", "현관", "기타 교외",
    "특별교실(과학실)", "교통구역(스쿨존 내)-인도", "학습지원공간", "기숙사",
    "기타 문화·체육공간", "공원, 유원 시설", "청소년 수련 시설", "어린이 놀이시설",
    "교통구역(스쿨존 외)-인도", "전시관, 체험관", "숙박시설/식당", "산림·계곡",
    "교통구역(스쿨존 내)-기타 교통구역", "교통구역(스쿨존 외)-기타 교통구역",
    "교통구역(스쿨존 내)-차도", "기타 자연", "현장실습/근로지(직업계고)",
    "교통구역(스쿨존 내)-자전거도로", "교통구역(스쿨존 외)-교통수단 안",
    "교통구역(스쿨존 외)-차도", "기타 관리·행정공간", "교통구역(스쿨존 내)-교통수단 안",
    "문화유적지", "기타 보건·위생공간", "교통구역(스쿨존 외)-자전거도로",
    "강·바다·하천", "영화관, 공연장", "탈의실/샤워실", "보건실", "가정",
    "교무실", "승강기", "행정실/방송실",
]

ACCIDENT_TYPES = [
    "넘어짐", "고정된 물체와의 부딪힘", "움직이는 물체와의 부딪힘",
    "스포츠 활동 중 충격을 가함", "사람과의 부딪힘", "그밖의 손상 사고",
    "이동 중 충격을 가함", "긁힘, 찔림", "1미터 미만의 높이에서 떨어짐",
    "물체 사이에 끼임·눌림", "베임, 절단", "고온의 물체·물질 접촉·흡입·섭취",
    "1미터 이상의 높이에서 떨어짐", "사람 사이에 끼임·눌림", "식중독",
    "물건을 운반하는 중 충격을 가함", "화학물질 접촉·흡입·섭취",
    "동물에게 물림(사람 포함)", "기타 호흡 곤란", "이물질 섭취로 인한 질병",
    "곤충·식물 등에 쏘임", "이물질 접촉에 의한 피부염", "교통사고", "감전",
    "일사병, 열사병", "이물질에 의한 질식",
    "저온의 물체(드라이아이스 등)·물질 접촉", "익사·익수", "추위에 장시간 노출",
]

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

# 사고유형 조합 총 개수 (학교급 6 × 장소 48 × 형태 29)
TOTAL_TYPES = len(SCHOOL_LEVELS) * len(PLACES) * len(ACCIDENT_TYPES)

# PLACES·ACCIDENT_TYPES 는 실제 발생 빈도 내림차순으로 정렬돼 있으므로,
# 순서를 그대로 '빈도 지표(0~1)'로 사용한다 (앞일수록 흔한 사고).
PLACE_FREQ = {p: 1 - i / (len(PLACES) - 1) for i, p in enumerate(PLACES)}
TYPE_FREQ = {t: 1 - i / (len(ACCIDENT_TYPES) - 1) for i, t in enumerate(ACCIDENT_TYPES)}


# ---------------------------------------------------------------------------
# 내부 유틸: 입력 조합을 재현 가능한 난수 시드로 변환 (mock 전용)
# ---------------------------------------------------------------------------
def _seed(*parts: str) -> int:
    key = "|".join(parts)
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)


def _stable_unit(key: str) -> float:
    """문자열 -> 0~1 사이의 재현 가능한 값 (지정 안 된 범주의 기본 가중치용)."""
    return (int(hashlib.md5(("w:" + key).encode()).hexdigest(), 16) % 10000) / 10000.0


def _weight(table: dict, key: str, lo: float, hi: float) -> float:
    """지정된 가중치가 있으면 사용, 없으면 이름 해시로 [lo, hi] 범위의 값을 부여."""
    if key in table:
        return table[key]
    return lo + _stable_unit(str(key)) * (hi - lo)


def _mock_analyze(school: str, place: str, acc_type: str) -> dict:
    """
    실제 모델이 없을 때 사용하는 재현 가능한 가상 추론.
    동일 입력 -> 동일 결과. 도메인 상식(운동장·넘어짐이 고위험)을 반영한다.
    """
    rng = np.random.default_rng(_seed(school, place, acc_type))

    # 도메인 가중치: 노트북 EDA 결론 반영. 주요 범주만 지정하고,
    # 나머지 48개 장소·29개 형태는 이름 해시로 안정적 가중치를 부여한다.
    place_w = {"운동장": 0.9, "강당(체육관)": 0.85, "실·내외 체육시설": 0.82,
               "기타 체육·집회공간": 0.78, "놀이터": 0.7, "어린이 놀이시설": 0.68,
               "계단": 0.6, "복도": 0.5, "일반(교과)교실": 0.42,
               "교통구역(스쿨존 내)-차도": 0.72, "교통구역(스쿨존 내)-인도": 0.55}
    type_w = {"넘어짐": 0.85, "움직이는 물체와의 부딪힘": 0.72,
              "고정된 물체와의 부딪힘": 0.7, "스포츠 활동 중 충격을 가함": 0.68,
              "사람과의 부딪힘": 0.6, "1미터 이상의 높이에서 떨어짐": 0.9,
              "1미터 미만의 높이에서 떨어짐": 0.68, "교통사고": 0.92, "감전": 0.95,
              "익사·익수": 0.98, "물체 사이에 끼임·눌림": 0.7, "베임, 절단": 0.6}
    level_w = {"중학교": 0.85, "고등학교": 0.78, "초등학교": 0.62,
               "유치원": 0.55, "특수학교": 0.7, "기타학교": 0.5}

    pw = _weight(place_w, place, 0.35, 0.78)
    tw = _weight(type_w, acc_type, 0.35, 0.78)
    lw = _weight(level_w, school, 0.5, 0.85)
    base = 0.4 * pw + 0.4 * tw + 0.2 * lw
    noise = rng.normal(0, 0.05)

    risk_prob = float(np.clip(base + noise, 0.02, 0.98))              # X1 위험확률
    # 예상 보상금: 고액사고(추락·교통·감전 등) 가중, 로그정규 분포
    sev_map = {"1미터 이상의 높이에서 떨어짐": 2.4, "교통사고": 2.3, "감전": 2.2,
               "익사·익수": 2.6, "1미터 미만의 높이에서 떨어짐": 1.6,
               "물체 사이에 끼임·눌림": 1.7, "베임, 절단": 1.5,
               "고정된 물체와의 부딪힘": 1.2, "움직이는 물체와의 부딪힘": 1.25,
               "넘어짐": 1.0, "그밖의 손상 사고": 1.3}
    sev_boost = _weight(sev_map, acc_type, 1.0, 1.5)
    comp = float(np.expm1(np.log1p(120_000) + risk_prob * 3.2) * sev_boost)  # X2 원
    comp = round(comp, -3)
    # 사고 빈도: 실제 발생 빈도 순서를 반영 (넘어짐·운동장이 압도적, 감전·익사는 희귀)
    freq_factor = PLACE_FREQ.get(place, 0.1) * TYPE_FREQ.get(acc_type, 0.1)
    freq = int(np.clip(rng.normal(20 + 1980 * freq_factor, 60), 3, 2000))  # X3 건

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
        {"요인": f"사고장소 = {place}", "기여도": round(pw * 0.18 + rng.normal(0, 0.01), 4)},
        {"요인": f"사고형태 = {acc_type}", "기여도": round(tw * 0.16 + rng.normal(0, 0.01), 4)},
        {"요인": f"학교급 = {school}", "기여도": round(lw * 0.11 + rng.normal(0, 0.01), 4)},
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
@lru_cache(maxsize=1)
def get_top10_priority() -> pd.DataFrame:
    """전체 사고유형 조합(수천 건)을 분석해 TOPSIS_CC 상위 10개를 반환한다(1회 캐시)."""
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
