"""
advisor.py
AI Safety Advisor.

입력(학교급·사고장소·사고형태·위험등급·예상보상금·사고빈도·예방우선순위)을 받아
시설관리·운영관리·안전교육을 자연스럽게 아우르는 **하나의 친절한 안내 문단**을 작성한다.

- 환경변수에 API 키/토큰이 있으면 해당 LLM으로 생성.
- 없으면 노트북 strategy_rules 기반의 규칙형 폴백으로 생성(오프라인에서도 동작).

generate_safety_advice() 는 (안내문, 사용모델표기) 튜플을 반환한다.
"""

from __future__ import annotations
import os
import json
import ssl
import urllib.request
import urllib.error

from backend import recommend_strategy


def _ssl_context() -> ssl.SSLContext:
    """macOS 파이썬의 인증서 미설치 이슈 대응: certifi가 있으면 사용."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _build_prompt(ctx: dict) -> str:
    return f"""너는 학교 안전을 함께 챙겨 주는 다정한 AI 안전 도우미야.
아래 분석 결과를 바탕으로, 학교 선생님·관리자에게 **친절하고 따뜻한 말투로**
예방 방법을 알려 줘.

[분석 결과]
- 학교급: {ctx['학교급']}
- 사고장소: {ctx['사고장소']}
- 사고형태: {ctx['사고형태']}
- 위험등급: {ctx['위험등급']}
- 예상 보상금: {ctx['예상보상금']:,}원
- 사고 빈도: {ctx['사고빈도']}건
- 예방 우선순위: 전체 {ctx.get('전체_사고유형수', '-')}개 중 {ctx['예방우선순위']}위

[작성 규칙]
- 항목이나 제목으로 나누지 말고, **하나로 이어지는 자연스러운 문단**으로 써 줘.
- 그 안에 시설 점검·정비, 운영·감독, 학생 안전교육 내용이 자연스럽게 녹아들게 해 줘.
- "~해요", "~하면 좋아요", "함께 ~해요" 처럼 다정하고 부드러운 말투로.
- 4~6문장, 300자 내외. 마지막은 격려의 한마디로 마무리해 줘.
- 반드시 아래 JSON 형식으로만 출력.
{{"조언": "여기에 문단 전체를 한 문자열로"}}
"""


def _parse_advice_json(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start:end + 1])
    advice = obj.get("조언") or obj.get("advice")
    return advice.strip() if isinstance(advice, str) and advice.strip() else None


def _from_github_models(ctx: dict) -> tuple[str, str] | None:
    """GitHub Models(OpenAI 호환 무료 추론). 반환: (안내문, 모델표기)"""
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    # 붙여넣기 과정에서 섞인 공백/따옴표 제거
    token = token.strip().strip('"').strip("'").strip()
    if not token:
        return None
    # HTTP 헤더는 ASCII만 허용 — 토큰에 비ASCII 문자가 있으면 호출 불가(잘못 붙여넣은 토큰)
    if not token.isascii():
        return None
    endpoint = os.getenv("GITHUB_MODELS_ENDPOINT",
                         "https://models.github.ai/inference")
    model = os.getenv("GITHUB_MODEL", "openai/gpt-4.1")
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "너는 다정한 학교 안전 도우미다. 반드시 JSON 객체만 출력한다."},
            {"role": "user", "content": _build_prompt(ctx)},
        ],
        "temperature": 0.6,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=40, context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        advice = _parse_advice_json(data["choices"][0]["message"]["content"].strip())
        return (advice, f"GitHub Models · {model}") if advice else None
    except Exception:
        # 네트워크/인증/인코딩 등 어떤 실패든 규칙기반으로 폴백(앱이 죽지 않게)
        return None


def _from_gemini(ctx: dict) -> tuple[str, str] | None:
    """Google Gemini REST. 반환: (안내문, 모델표기)"""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    payload = {
        "contents": [{"parts": [{"text": _build_prompt(ctx)}]}],
        "generationConfig": {"temperature": 0.6,
                             "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        advice = _parse_advice_json(
            data["candidates"][0]["content"]["parts"][0]["text"].strip())
        return (advice, f"Google Gemini · {model}") if advice else None
    except Exception:
        return None


def _from_openai(ctx: dict) -> tuple[str, str] | None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_prompt(ctx)}],
            response_format={"type": "json_object"},
            temperature=0.6,
        )
        advice = _parse_advice_json(resp.choices[0].message.content)
        return (advice, f"OpenAI · {model}") if advice else None
    except Exception:
        return None


def _from_anthropic(ctx: dict) -> tuple[str, str] | None:
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user",
                       "content": _build_prompt(ctx) + "\n반드시 JSON 객체만 출력."}],
        )
        advice = _parse_advice_json(msg.content[0].text.strip())
        return (advice, f"Anthropic · {model}") if advice else None
    except Exception:
        return None


def _fallback(ctx: dict) -> tuple[str, str]:
    """LLM 미사용 시 strategy_rules 기반 규칙형 문단 생성(친절한 말투)."""
    school, place, atype = ctx["학교급"], ctx["사고장소"], ctx["사고형태"]
    high = ctx.get("위험등급_index", 3) <= 1

    place_s = recommend_strategy("사고장소", place).split(",")[0].strip()
    type_s = recommend_strategy("사고형태", atype).split(",")[0].strip()
    level_s = recommend_strategy("학교급", school)

    urgency = "특히 위험등급이 높은 만큼 조금 더 세심한 관심이 필요해요. " if high else ""

    advice = (
        f"{school} {place}에서 일어나는 '{atype}' 사고를 줄이려면, "
        f"먼저 {place}의 바닥과 시설물, 안전장비를 꼼꼼히 살펴보고 "
        f"{place_s} 같은 물리적 위험요인을 미리 없애 두면 좋아요. "
        f"{atype}이(가) 자주 생기는 쉬는시간·체육시간에는 순찰과 감독 인력을 배치해 "
        f"{type_s}에 신경 써 주시고요. {urgency}"
        f"학생들에게는 {level_s.rstrip('.')}과(와) 함께 활동 전 안전수칙을 "
        f"눈높이에 맞게 반복해서 알려 주면 큰 도움이 돼요. "
        f"작은 점검 하나가 소중한 학생을 지켜 줄 거예요. 함께 안전한 학교를 만들어가요! 😊"
    )
    return advice, "규칙 기반(오프라인 폴백)"


def generate_safety_advice(ctx: dict) -> tuple[str, str]:
    """
    Returns (advice_text, model_label)
      advice_text : 시설·운영·교육을 아우르는 친절한 단일 문단
      model_label : 실제 사용한 AI 모델 표기
                    예) "GitHub Models · openai/gpt-4.1", "규칙 기반(오프라인 폴백)"
    """
    if os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"):
        res = _from_github_models(ctx)
        if res:
            return res
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        res = _from_gemini(ctx)
        if res:
            return res
    if os.getenv("OPENAI_API_KEY"):
        res = _from_openai(ctx)
        if res:
            return res
    if os.getenv("ANTHROPIC_API_KEY"):
        res = _from_anthropic(ctx)
        if res:
            return res
    return _fallback(ctx)
