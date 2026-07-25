"""
backend.py
학교안전사고 예방 의사결정 지원 시스템의 분석 백엔드 (실제 학습 모델 연결판).

train_export.py 가 실제 데이터(accident.xlsx / compensation.xlsx)로 학습·산출해
저장한 model_artifacts.pkl 을 로드하여 추론한다.
  - 위험등급 분류(LightGBM, 6클래스)       -> 위험등급 / 위험확률(X1)
  - 예상 보상금(실측 평균보상액)             -> 예상 보상금(X2)
  - 사고 빈도(accident 실측, 연평균)          -> 사고 빈도(X3)
  - CRITIC-TOPSIS                            -> 예방 우선순위(TOPSIS_CC)
  - SHAP(TreeExplainer, 사전계산)            -> 위험도 상승 기여 요인

app.py 는 analyze_accident() 의 "키 계약(contract)"에만 의존하므로 수정 불필요.
model_artifacts.pkl 이 없으면 재현 가능한 mock 으로 자동 폴백한다(데모 안전장치).

★ 재학습/아티팩트 재생성:  python train_export.py
"""

from __future__ import annotations

import os
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

# 위험등급 라벨 (6단계 타깃, index 0 = 가장 위험)
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

_ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "model_artifacts.pkl")
FEATURE_COLS = ["학교급", "사고장소", "사고형태"]


# ---------------------------------------------------------------------------
# 아티팩트 로드 (1회 캐시)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_artifacts():
    """model_artifacts.pkl 을 로드한다. 없거나 실패하면 None 을 반환(→ mock 폴백)."""
    try:
        import joblib
        art = joblib.load(_ARTIFACT_PATH)
        return art
    except Exception as e:  # noqa: BLE001
        print(f"[backend] 아티팩트 로드 실패 → mock 폴백: {e}")
        return None


def _grade_index(grade: str) -> int:
    return RISK_GRADES.index(grade) if grade in RISK_GRADES else 3


def _shap_list(school, place, acc_type, s_school, s_place, s_type) -> list:
    shap = [
        {"요인": f"사고장소 = {place}", "기여도": round(float(s_place), 4)},
        {"요인": f"사고형태 = {acc_type}", "기여도": round(float(s_type), 4)},
        {"요인": f"학교급 = {school}", "기여도": round(float(s_school), 4)},
    ]
    shap.sort(key=lambda d: d["기여도"], reverse=True)
    return shap


def _encode_row(art: dict, school: str, place: str, acc_type: str):
    """encoders_map(dict) 로 3개 피처를 코드로 변환. 미등록 범주는 0 으로 폴백."""
    m = art["encoders_map"]
    import pandas as pd  # 지역 import (런타임 pandas 사용)
    return pd.DataFrame(
        [[m["학교급"].get(school, 0), m["사고장소"].get(place, 0),
          m["사고형태"].get(acc_type, 0)]], columns=FEATURE_COLS)


def _real_analyze(art: dict, school: str, place: str, acc_type: str) -> dict:
    recs = art["table_records"]
    lookup = art["lookup"]
    total = art["total_types"]
    key = (school, place, acc_type)

    if key in lookup:
        # ---- 학습된 조합: 사전계산 결과 그대로 사용 ----
        row = recs[lookup[key]]
        grade = str(row["위험등급"])
        risk_prob = float(row["X1_위험확률"])
        comp = int(row["X2_예상보상금"])
        freq = int(row["X3_사고빈도"])
        topsis_cc = float(row["TOPSIS_CC"])
        rank = int(row["순위"])
        hi2 = art["norm_ranges"]["X2_예상보상금"][1]
        norm = {
            "위험확률": risk_prob,
            # 표시용: 예상보상금은 이상치로 인한 선형 min-max 눌림을 피해 로그 스케일 사용
            # (TOPSIS_CC 계산에는 영향 없음 — 아티팩트에 이미 계산돼 저장됨)
            "예상보상금": float(np.clip(np.log1p(comp) / np.log1p(hi2), 0, 1)) if hi2 > 0 else 0.0,
            "사고빈도": float(row["X3_사고빈도_n"]),
            "TOPSIS": topsis_cc,
        }
        shap = _shap_list(school, place, acc_type,
                          row["shap_학교급"], row["shap_사고장소"], row["shap_사고형태"])
    else:
        # ---- 미출현 조합: 실시간 모델 추론 + 데이터 기반 폴백 ----
        clf = art["clf"]
        top_col = art["top_class_col"]
        x = _encode_row(art, school, place, acc_type)
        proba = clf.predict_proba(x)[0]
        risk_prob = float(proba[top_col])
        grade = RISK_GRADES[int(clf.predict(x)[0])]

        comp = int(art["comp_type_mean"].get(acc_type, art["comp_global"]))
        freq = 0  # 관측된 발생기록 없음

        lo1, hi1 = art["norm_ranges"]["X1_위험확률"]
        lo2, hi2 = art["norm_ranges"]["X2_예상보상금"]
        lo3, hi3 = art["norm_ranges"]["X3_사고빈도"]
        x1n = np.clip((risk_prob - lo1) / (hi1 - lo1) if hi1 > lo1 else 0, 0, 1)
        x2n = np.clip((comp - lo2) / (hi2 - lo2) if hi2 > lo2 else 0, 0, 1)
        x3n = np.clip((freq - lo3) / (hi3 - lo3) if hi3 > lo3 else 0, 0, 1)
        w = art["critic_weights"]
        cc_proxy = float(w["X1_위험확률"] * x1n + w["X2_예상보상금"] * x2n
                         + w["X3_사고빈도"] * x3n)
        topsis_cc = cc_proxy
        rank = int((art["all_cc"] > cc_proxy).sum()) + 1
        comp_disp = float(np.clip(np.log1p(comp) / np.log1p(hi2), 0, 1)) if hi2 > 0 else 0.0
        norm = {"위험확률": risk_prob, "예상보상금": comp_disp,
                "사고빈도": float(x3n), "TOPSIS": topsis_cc}

        cms = art["cat_mean_shap"]
        shap = _shap_list(
            school, place, acc_type,
            cms["학교급"].get(school, 0.0),
            cms["사고장소"].get(place, 0.0),
            cms["사고형태"].get(acc_type, 0.0),
        )

    return {
        "위험등급": grade,
        "위험등급_index": _grade_index(grade),
        "위험확률": risk_prob,
        "예상보상금": comp,
        "사고빈도": freq,
        "TOPSIS_CC": topsis_cc,
        "예방우선순위": rank,
        "전체_사고유형수": total,
        "지표_정규화": norm,
        "shap": shap,
    }


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def analyze_accident(school_level: str, place: str, accident_type: str) -> dict:
    """
    학교급·사고장소·사고형태를 입력받아 예방 의사결정 지표를 반환한다.

    Returns (키 계약 — app.py 가 의존):
        위험등급, 위험등급_index, 위험확률(0~1), 예상보상금(원), 사고빈도(건),
        TOPSIS_CC(0~1), 예방우선순위(1=최우선), 전체_사고유형수,
        지표_정규화{위험확률, 예상보상금, 사고빈도, TOPSIS}, shap[{요인, 기여도}]
    """
    art = _load_artifacts()
    if art is not None:
        return _real_analyze(art, school_level, place, accident_type)
    return _mock_analyze(school_level, place, accident_type)  # 폴백


@lru_cache(maxsize=1)
def get_top10_priority() -> pd.DataFrame:
    """TOPSIS_CC 상위 10개 사고유형을 반환한다(1회 캐시)."""
    art = _load_artifacts()
    if art is not None:
        recs = sorted(art["table_records"], key=lambda r: r["TOPSIS_CC"], reverse=True)[:10]
        out = pd.DataFrame([{
            "순위": int(r["순위"]),
            "사고유형": r["사고유형"],
            "학교급": r["학교급"],
            "사고장소": r["사고장소"],
            "사고형태": r["사고형태"],
            "위험등급": r["위험등급"],
            "위험확률": float(r["X1_위험확률"]),
            "예상보상금": int(r["X2_예상보상금"]),
            "사고빈도": int(r["X3_사고빈도"]),
            "TOPSIS_CC": float(r["TOPSIS_CC"]),
        } for r in recs])
        return out.reset_index(drop=True)
    return _mock_top10()  # 폴백


def recommend_strategy(feature: str, category: str) -> str:
    default = "현장 점검과 맞춤형 안전교육을 강화"
    for keyword, strategy in STRATEGY_RULES.get(feature, {}).items():
        if keyword in str(category):
            return strategy
    return default


# ===========================================================================
# 폴백(mock): model_artifacts.pkl 이 없을 때만 사용. 재현 가능한 가상 추론.
# ===========================================================================
TOTAL_TYPES = len(SCHOOL_LEVELS) * len(PLACES) * len(ACCIDENT_TYPES)
PLACE_FREQ = {p: 1 - i / (len(PLACES) - 1) for i, p in enumerate(PLACES)}
TYPE_FREQ = {t: 1 - i / (len(ACCIDENT_TYPES) - 1) for i, t in enumerate(ACCIDENT_TYPES)}


def _seed(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode()).hexdigest(), 16) % (2**32)


def _stable_unit(key: str) -> float:
    return (int(hashlib.md5(("w:" + key).encode()).hexdigest(), 16) % 10000) / 10000.0


def _weight(table: dict, key: str, lo: float, hi: float) -> float:
    if key in table:
        return table[key]
    return lo + _stable_unit(str(key)) * (hi - lo)


def _mock_analyze(school: str, place: str, acc_type: str) -> dict:
    rng = np.random.default_rng(_seed(school, place, acc_type))
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
    risk_prob = float(np.clip(base + rng.normal(0, 0.05), 0.02, 0.98))
    sev_map = {"1미터 이상의 높이에서 떨어짐": 2.4, "교통사고": 2.3, "감전": 2.2,
               "익사·익수": 2.6, "1미터 미만의 높이에서 떨어짐": 1.6,
               "물체 사이에 끼임·눌림": 1.7, "베임, 절단": 1.5,
               "고정된 물체와의 부딪힘": 1.2, "움직이는 물체와의 부딪힘": 1.25,
               "넘어짐": 1.0, "그밖의 손상 사고": 1.3}
    sev_boost = _weight(sev_map, acc_type, 1.0, 1.5)
    comp = round(float(np.expm1(np.log1p(120_000) + risk_prob * 3.2) * sev_boost), -3)
    freq_factor = PLACE_FREQ.get(place, 0.1) * TYPE_FREQ.get(acc_type, 0.1)
    freq = int(np.clip(rng.normal(20 + 1980 * freq_factor, 60), 3, 2000))
    thr = [0.85, 0.72, 0.58, 0.45, 0.30]
    grade_idx = int(np.digitize(-risk_prob, [-t for t in thr]))
    norm = {"위험확률": risk_prob,
            "예상보상금": float(np.clip(np.log1p(comp) / np.log1p(50_000_000), 0, 1)),
            "사고빈도": float(np.clip(freq / 2000, 0, 1))}
    topsis_cc = float(np.clip(0.45 * norm["위험확률"] + 0.30 * norm["예상보상금"]
                              + 0.25 * norm["사고빈도"] + rng.normal(0, 0.02), 0.02, 0.98))
    norm["TOPSIS"] = topsis_cc
    rank = int(round((1 - topsis_cc) * (TOTAL_TYPES - 1))) + 1
    shap = [
        {"요인": f"사고장소 = {place}", "기여도": round(pw * 0.18 + rng.normal(0, 0.01), 4)},
        {"요인": f"사고형태 = {acc_type}", "기여도": round(tw * 0.16 + rng.normal(0, 0.01), 4)},
        {"요인": f"학교급 = {school}", "기여도": round(lw * 0.11 + rng.normal(0, 0.01), 4)},
    ]
    shap.sort(key=lambda d: d["기여도"], reverse=True)
    return {"위험등급": RISK_GRADES[grade_idx], "위험등급_index": grade_idx,
            "위험확률": risk_prob, "예상보상금": int(comp), "사고빈도": freq,
            "TOPSIS_CC": topsis_cc, "예방우선순위": rank, "전체_사고유형수": TOTAL_TYPES,
            "지표_정규화": norm, "shap": shap}


def _mock_top10() -> pd.DataFrame:
    rows = []
    for s in SCHOOL_LEVELS:
        for p in PLACES:
            for t in ACCIDENT_TYPES:
                r = _mock_analyze(s, p, t)
                rows.append({"사고유형": f"{s}-{p}-{t}", "학교급": s, "사고장소": p,
                             "사고형태": t, "위험등급": r["위험등급"],
                             "위험확률": r["위험확률"], "예상보상금": r["예상보상금"],
                             "사고빈도": r["사고빈도"], "TOPSIS_CC": r["TOPSIS_CC"]})
    df = pd.DataFrame(rows).sort_values("TOPSIS_CC", ascending=False).reset_index(drop=True)
    df.insert(0, "순위", df.index + 1)
    return df.head(10)
