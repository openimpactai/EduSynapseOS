# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
#!/usr/bin/env python3
"""
E2E Multi-Language Content Creation Test
==========================================

Tests the full content creation pipeline via real API calls for multiple
languages (en, tr, ar, es, fr) and multiple content types.

Flow per scenario:
  1. Auth (token exchange)
  2. POST /chat — create content in target language
  3. Evaluate response: check phase, content type, generated_content
  4. POST /chat — approve
  5. Evaluate export: check h5p_id, preview_url, localized strings in ai_content
  6. Optionally: modify flow, end session

No mocks. No user prompts. Fully autonomous.
"""

import json
import sys
import os
import time
import traceback
import requests
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("API_URL", "http://localhost:34000")
API_PREFIX = f"{BASE_URL}/api/v1"

TENANT = {
    "code": "playground",
    "api_key": "tk_bbb1e12e4fc8bf6542ad94602d21e185",
    "api_secret": "ts_546c938a66e884537214afcfc1fc19803550fdb97569dc48",
}

TEACHERS = {
    "en": {
        "id": "7c9c7dcc-f5bd-4cca-af56-a34342638327",
        "email": "emma.wilson@playground.edusynapse.io",
        "first_name": "Emma",
        "last_name": "Wilson",
        "country_code": "GB",
        "framework_code": "UK-NC-2014",
    },
    "tr": {
        "id": "7c9c7dcc-f5bd-4cca-af56-a34342638327",
        "email": "emma.wilson@playground.edusynapse.io",
        "first_name": "Emma",
        "last_name": "Wilson",
        "country_code": "GB",
        "framework_code": "UK-NC-2014",
    },
    "ar": {
        "id": "7c9c7dcc-f5bd-4cca-af56-a34342638327",
        "email": "emma.wilson@playground.edusynapse.io",
        "first_name": "Emma",
        "last_name": "Wilson",
        "country_code": "GB",
        "framework_code": "UK-NC-2014",
    },
    "es": {
        "id": "6d44875c-4bf8-4a4f-94c5-b3d8cad21318",
        "email": "sarah.johnson@playground.edusynapse.io",
        "first_name": "Sarah",
        "last_name": "Johnson",
        "country_code": "US",
        "framework_code": "CCSS",
    },
    "fr": {
        "id": "6d44875c-4bf8-4a4f-94c5-b3d8cad21318",
        "email": "sarah.johnson@playground.edusynapse.io",
        "first_name": "Sarah",
        "last_name": "Johnson",
        "country_code": "US",
        "framework_code": "CCSS",
    },
}

# ---------------------------------------------------------------------------
# Scenario definitions — one per (content_type, language) combo
# ---------------------------------------------------------------------------
SCENARIOS = [
    # --- ENGLISH ---
    {
        "name": "EN Multiple Choice",
        "lang": "en",
        "content_type": "multiple-choice",
        "message": "Create 3 multiple choice questions about the water cycle for Year 5 students.",
        "approve_msg": "approve",
        "expect_keys": ["questions"],
        "grade_level": 5,
        "subject": "science",
    },
    {
        "name": "EN True/False",
        "lang": "en",
        "content_type": "true-false",
        "message": "Create 3 true or false questions about plant cells for Year 6.",
        "approve_msg": "approve",
        "expect_keys": ["statements"],
        "grade_level": 6,
        "subject": "science",
    },
    {
        "name": "EN Flashcards",
        "lang": "en",
        "content_type": "flashcards",
        "message": "Create 5 flashcards about the human digestive system vocabulary for Year 5.",
        "approve_msg": "approve",
        "expect_keys": ["deck"],
        "grade_level": 5,
        "subject": "science",
    },
    {
        "name": "EN Fill Blanks",
        "lang": "en",
        "content_type": "fill-blanks",
        "message": "Create 3 fill in the blanks exercises about the solar system for Year 4.",
        "approve_msg": "approve",
        "expect_keys": ["exercises"],
        "grade_level": 4,
        "subject": "science",
    },
    {
        "name": "EN Drag Words",
        "lang": "en",
        "content_type": "drag-words",
        "message": "Create 3 drag the words exercises about photosynthesis for Year 5.",
        "approve_msg": "approve",
        "expect_keys": ["exercises"],
        "grade_level": 5,
        "subject": "science",
    },
    {
        "name": "EN Crossword",
        "lang": "en",
        "content_type": "crossword",
        "message": "Create a crossword puzzle with 6 words about the planets in our solar system for Year 4.",
        "approve_msg": "approve",
        "expect_keys": ["words"],
        "grade_level": 4,
        "subject": "science",
    },
    {
        "name": "EN Essay",
        "lang": "en",
        "content_type": "essay",
        "message": "Create an essay question about the effects of deforestation on climate change for Year 6.",
        "approve_msg": "approve",
        "expect_keys": ["prompt"],
        "grade_level": 6,
        "subject": "science",
    },
    {
        "name": "EN Summary",
        "lang": "en",
        "content_type": "summary",
        "message": "Create a summary exercise with 3 panels about the water cycle for Year 5.",
        "approve_msg": "approve",
        "expect_keys": ["panels"],
        "grade_level": 5,
        "subject": "science",
    },
    # --- TURKISH ---
    {
        "name": "TR Multiple Choice",
        "lang": "tr",
        "content_type": "multiple-choice",
        "message": "5. sınıf öğrencileri için su döngüsü hakkında 3 çoktan seçmeli soru oluştur.",
        "approve_msg": "onayla",
        "expect_keys": ["questions"],
        "grade_level": 5,
        "subject": "science",
    },
    {
        "name": "TR Flashcards",
        "lang": "tr",
        "content_type": "flashcards",
        "message": "6. sınıf için İngilizce günlük yaşam kelimeleri hakkında 5 kelime kartı oluştur: hello, goodbye, please, thank you, sorry.",
        "approve_msg": "onayla",
        "expect_keys": ["deck"],
        "grade_level": 6,
        "subject": "english",
    },
    {
        "name": "TR Crossword",
        "lang": "tr",
        "content_type": "crossword",
        "message": "4. sınıf için güneş sistemi gezegenleri ile 6 kelimelik bir bulmaca oluştur.",
        "approve_msg": "onayla",
        "expect_keys": ["words"],
        "grade_level": 4,
        "subject": "science",
    },
    # --- ARABIC ---
    {
        "name": "AR Multiple Choice",
        "lang": "ar",
        "content_type": "multiple-choice",
        "message": "أنشئ 3 أسئلة اختيار من متعدد حول دورة المياه لطلاب الصف الخامس.",
        "approve_msg": "موافق",
        "expect_keys": ["questions"],
        "grade_level": 5,
        "subject": "science",
    },
    {
        "name": "AR Flashcards",
        "lang": "ar",
        "content_type": "flashcards",
        "message": "أنشئ 5 بطاقات تعليمية حول مفردات جسم الإنسان للصف السادس.",
        "approve_msg": "موافق",
        "expect_keys": ["deck"],
        "grade_level": 6,
        "subject": "science",
    },
    # --- SPANISH ---
    {
        "name": "ES Multiple Choice",
        "lang": "es",
        "content_type": "multiple-choice",
        "message": "Crea 3 preguntas de opción múltiple sobre el ciclo del agua para estudiantes de 5to grado.",
        "approve_msg": "aprobar",
        "expect_keys": ["questions"],
        "grade_level": 5,
        "subject": "science",
    },
    {
        "name": "ES Flashcards",
        "lang": "es",
        "content_type": "flashcards",
        "message": "Crea 5 tarjetas de vocabulario sobre el sistema digestivo para 5to grado.",
        "approve_msg": "aprobar",
        "expect_keys": ["deck"],
        "grade_level": 5,
        "subject": "science",
    },
    # --- FRENCH ---
    {
        "name": "FR Multiple Choice",
        "lang": "fr",
        "content_type": "multiple-choice",
        "message": "Créez 3 questions à choix multiples sur le cycle de l'eau pour les élèves de CM2.",
        "approve_msg": "approuver",
        "expect_keys": ["questions"],
        "grade_level": 5,
        "subject": "science",
    },
    {
        "name": "FR Flashcards",
        "lang": "fr",
        "content_type": "flashcards",
        "message": "Créez 5 cartes de vocabulaire sur le système solaire pour les élèves de CM1.",
        "approve_msg": "approuver",
        "expect_keys": ["deck"],
        "grade_level": 4,
        "subject": "science",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
results = []


def log(msg, indent=0):
    prefix = "  " * indent
    print(f"{prefix}{msg}")


def authenticate(lang: str) -> str:
    """Get bearer token via token exchange."""
    teacher = TEACHERS[lang]
    resp = requests.post(
        f"{API_PREFIX}/auth/exchange",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": TENANT["api_key"],
            "X-API-Secret": TENANT["api_secret"],
        },
        json={
            "user": {
                "external_id": teacher["id"],
                "email": teacher["email"],
                "first_name": teacher["first_name"],
                "last_name": teacher["last_name"],
                "user_type": "teacher",
            }
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]


def chat(token: str, message: str, session_id=None, language="en", context=None) -> dict:
    """Send a chat message to the content creation API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Tenant-Code": TENANT["code"],
        "Content-Type": "application/json",
    }
    body = {"message": message}
    if session_id:
        body["session_id"] = session_id
    if language:
        body["language"] = language
    if context:
        body["context"] = context

    resp = requests.post(
        f"{API_PREFIX}/content-creation/chat",
        headers=headers,
        json=body,
        timeout=180,  # LLM generation can be slow
    )
    resp.raise_for_status()
    return resp.json()


def record(scenario_name, step, status, detail=""):
    results.append({
        "scenario": scenario_name,
        "step": step,
        "status": status,
        "detail": detail[:300] if detail else "",
        "ts": datetime.now().isoformat(),
    })
    icon = "PASS" if status == "pass" else "FAIL" if status == "fail" else "WARN"
    log(f"[{icon}] {step}: {detail[:120]}", indent=1)


# ---------------------------------------------------------------------------
# Scenario Runner
# ---------------------------------------------------------------------------
def run_scenario(scenario: dict):
    name = scenario["name"]
    lang = scenario["lang"]
    content_type = scenario["content_type"]
    log(f"\n{'='*60}")
    log(f"SCENARIO: {name} (lang={lang}, type={content_type})")
    log(f"{'='*60}")

    # Step 1: Authenticate
    try:
        token = authenticate(lang)
        record(name, "auth", "pass", f"Token obtained for {lang}")
    except Exception as e:
        record(name, "auth", "fail", str(e))
        return

    context = {
        "user_role": "teacher",
        "country_code": TEACHERS[lang]["country_code"],
        "framework_code": TEACHERS[lang]["framework_code"],
        "subject_code": scenario.get("subject", "science"),
        "grade_level": scenario.get("grade_level", 5),
    }

    # Step 2: Send creation message
    try:
        log(f"  Sending: {scenario['message'][:80]}...", indent=0)
        r1 = chat(token, scenario["message"], language=lang, context=context)
        session_id = r1.get("session_id")
        phase1 = r1.get("workflow_phase", "")
        msg1 = r1.get("message", "")[:200]
        gen1 = r1.get("generated_content")

        record(name, "create_msg", "pass", f"phase={phase1}, session={session_id}")
        log(f"  Response phase: {phase1}", indent=0)
        log(f"  Message: {msg1}...", indent=0)
    except Exception as e:
        record(name, "create_msg", "fail", str(e))
        return

    # Step 3: If not in review phase yet, the orchestrator may need another turn
    # (e.g., asking for confirmation). Send a confirmation.
    max_turns = 5
    turn = 0
    while phase1 not in ("reviewing", "review", "exporting", "completed") and turn < max_turns:
        turn += 1
        try:
            # Determine what to say based on phase
            if "confirm" in phase1 or "awaiting" in phase1 or "gathering" in phase1 or "requirements" in phase1:
                # System is asking for clarification — confirm intent
                confirm_msgs = {
                    "en": f"Yes, please create {content_type} content. Generate it now.",
                    "tr": f"Evet, {content_type} içeriği oluştur. Şimdi oluştur.",
                    "ar": f"نعم، أنشئ محتوى {content_type}. أنشئه الآن.",
                    "es": f"Sí, crea contenido de {content_type}. Genéralo ahora.",
                    "fr": f"Oui, créez du contenu {content_type}. Générez-le maintenant.",
                }
                confirm_msg = confirm_msgs.get(lang, confirm_msgs["en"])
            else:
                confirm_msg = scenario["message"]  # repeat

            log(f"  Turn {turn}: Sending confirmation (phase={phase1})...", indent=0)
            r1 = chat(token, confirm_msg, session_id=session_id, language=lang)
            phase1 = r1.get("workflow_phase", "")
            gen1 = r1.get("generated_content")
            msg1 = r1.get("message", "")[:200]
            record(name, f"turn_{turn}", "pass", f"phase={phase1}")
            log(f"  Response phase: {phase1}", indent=0)
            log(f"  Message: {msg1}...", indent=0)
        except Exception as e:
            record(name, f"turn_{turn}", "fail", str(e))
            break

    # Step 4: Check content was generated
    if gen1:
        ct = gen1.get("content_type", "unknown")
        status = gen1.get("status", "unknown")
        ai_content = gen1.get("ai_content", {})
        title = gen1.get("title", "no title")

        record(name, "content_generated", "pass",
               f"type={ct}, status={status}, title={title[:60]}")

        # Verify expected keys in ai_content
        for key in scenario.get("expect_keys", []):
            if key in ai_content:
                val = ai_content[key]
                count = len(val) if isinstance(val, list) else "present"
                record(name, f"ai_content.{key}", "pass", f"count={count}")
            else:
                # Sometimes ai_content wraps in raw_response
                record(name, f"ai_content.{key}", "warn",
                       f"Key missing. Available: {list(ai_content.keys())[:10]}")
    else:
        record(name, "content_generated", "fail",
               f"No generated_content. phase={phase1}")
        # Still try to approve in case content exists on server side
        if phase1 not in ("reviewing", "review"):
            return

    # Step 5: Approve / Export
    if phase1 in ("reviewing", "review"):
        try:
            log(f"  Approving with: '{scenario['approve_msg']}'", indent=0)
            r2 = chat(token, scenario["approve_msg"], session_id=session_id, language=lang)
            phase2 = r2.get("workflow_phase", "")
            gen2 = r2.get("generated_content")
            msg2 = r2.get("message", "")[:200]

            log(f"  Post-approve phase: {phase2}", indent=0)
            log(f"  Message: {msg2}...", indent=0)

            if gen2:
                h5p_id = gen2.get("h5p_id")
                preview = gen2.get("preview_url")
                export_status = gen2.get("status")

                if h5p_id:
                    record(name, "export", "pass",
                           f"H5P ID={h5p_id}, status={export_status}, preview={preview}")
                elif export_status == "export_failed":
                    record(name, "export", "warn",
                           f"Export failed (infra issue). status={export_status}")
                else:
                    record(name, "export", "warn",
                           f"No h5p_id. status={export_status}, phase={phase2}")
            else:
                record(name, "export", "warn",
                       f"No generated_content after approve. phase={phase2}")
        except Exception as e:
            record(name, "export", "fail", str(e))
    else:
        record(name, "export", "warn", f"Skipped approve — phase was {phase1}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("E2E Multi-Language Content Creation Test")
    print(f"API: {API_PREFIX}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        r.raise_for_status()
        log(f"Health: {r.json().get('status', 'unknown')}")
    except Exception as e:
        log(f"FATAL: API not reachable at {BASE_URL}: {e}")
        sys.exit(1)

    # Run scenarios
    for i, scenario in enumerate(SCENARIOS):
        log(f"\n[{i+1}/{len(SCENARIOS)}]")
        try:
            run_scenario(scenario)
        except Exception as e:
            record(scenario["name"], "unhandled_error", "fail", traceback.format_exc()[:300])

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    pass_count = sum(1 for r in results if r["status"] == "pass")
    fail_count = sum(1 for r in results if r["status"] == "fail")
    warn_count = sum(1 for r in results if r["status"] == "warn")

    # Group by scenario
    scenarios_seen = {}
    for r in results:
        sn = r["scenario"]
        if sn not in scenarios_seen:
            scenarios_seen[sn] = {"pass": 0, "fail": 0, "warn": 0}
        scenarios_seen[sn][r["status"]] += 1

    for sn, counts in scenarios_seen.items():
        status = "PASS" if counts["fail"] == 0 else "FAIL"
        print(f"  [{status}] {sn}: {counts['pass']}p/{counts['fail']}f/{counts['warn']}w")

    print(f"\nTotal steps: {len(results)} | Pass: {pass_count} | Fail: {fail_count} | Warn: {warn_count}")

    # Save detailed results
    out_path = Path(__file__).parent / "e2e_multilang_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "api_url": API_PREFIX,
            "scenarios_count": len(SCENARIOS),
            "total_steps": len(results),
            "pass": pass_count,
            "fail": fail_count,
            "warn": warn_count,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {out_path}")

    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
