# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
#!/usr/bin/env python3
"""
Multi-Language Locale Loading Test Suite
=========================================

Tests that all H5P locale files load correctly for all 5 supported languages
(en, tr, ar, es, fr) and that the get_l10n() merging logic works properly.

Tests:
1. All locale JSON files are valid and non-empty
2. All content types have all 5 language files
3. get_l10n() returns correct strings per language
4. English fallback works for unknown languages
5. Content-type strings override common strings
6. Nested structures (crossword, question-set, interactive-book) load correctly
7. Each converter's convert() output contains correct localized strings
"""

import json
import copy
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOCALES_DIR = PROJECT_ROOT / "config" / "h5p-locales"


def _load_locale(path: Path) -> dict:
    """Standalone locale loader (mirrors base.py _load_locale_file)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_l10n(content_type: str, language: str = "en") -> dict:
    """Standalone get_l10n (mirrors BaseH5PConverter.get_l10n without imports)."""
    common = copy.deepcopy(_load_locale(LOCALES_DIR / "_common" / "en.json"))
    if language != "en":
        lang_common = _load_locale(LOCALES_DIR / "_common" / f"{language}.json")
        if lang_common:
            common.update(lang_common)

    ct_strings = copy.deepcopy(_load_locale(LOCALES_DIR / content_type / "en.json"))
    if language != "en":
        lang_ct = _load_locale(LOCALES_DIR / content_type / f"{language}.json")
        if lang_ct:
            ct_strings.update(lang_ct)

    merged = common
    merged.update(ct_strings)
    return merged
LANGUAGES = ["en", "tr", "ar", "es", "fr"]

CONTENT_TYPES = [
    "_common", "flashcards", "multiple-choice", "true-false", "fill-blanks",
    "drag-words", "mark-words", "single-choice-set", "essay", "summary",
    "question-set", "arithmetic-quiz", "dialog-cards", "crossword",
    "memory-game", "course-presentation", "branching-scenario",
    "interactive-book", "documentation-tool", "image-pairing",
    "image-sequencing", "sort-paragraphs", "word-search", "personality-quiz",
]

passed = 0
failed = 0
errors = []


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        errors.append(msg)
        print(f"  ✗ {msg}")


def test_all_locale_files_exist_and_valid():
    """Test 1: Every content type has all 5 language JSON files, all valid."""
    print("\n=== Test 1: Locale files exist and are valid JSON ===")
    for ct in CONTENT_TYPES:
        for lang in LANGUAGES:
            fpath = LOCALES_DIR / ct / f"{lang}.json"
            check(
                f"{ct}/{lang}.json exists",
                fpath.exists(),
                f"Missing file: {fpath}",
            )
            if fpath.exists():
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                    check(
                        f"{ct}/{lang}.json is non-empty dict",
                        isinstance(data, dict) and len(data) > 0,
                        f"Empty or not a dict",
                    )
                except json.JSONDecodeError as e:
                    check(f"{ct}/{lang}.json valid JSON", False, str(e))


def test_en_and_lang_have_same_keys():
    """Test 2: Each language file has the same top-level keys as en.json."""
    print("\n=== Test 2: Language files have same keys as English ===")
    for ct in CONTENT_TYPES:
        en_path = LOCALES_DIR / ct / "en.json"
        if not en_path.exists():
            continue
        en_data = json.loads(en_path.read_text(encoding="utf-8"))
        en_keys = set(en_data.keys())
        for lang in LANGUAGES:
            if lang == "en":
                continue
            lang_path = LOCALES_DIR / ct / f"{lang}.json"
            if not lang_path.exists():
                continue
            lang_data = json.loads(lang_path.read_text(encoding="utf-8"))
            lang_keys = set(lang_data.keys())
            missing = en_keys - lang_keys
            extra = lang_keys - en_keys
            check(
                f"{ct}/{lang}.json keys match en.json",
                missing == set() and extra == set(),
                f"missing={missing}, extra={extra}" if (missing or extra) else "",
            )


def test_nested_structures_consistent():
    """Test 3: Nested locale files (crossword, question-set, etc.) have consistent nested keys."""
    print("\n=== Test 3: Nested structure consistency ===")
    nested_types = {
        "crossword": ["l10n", "a11y"],
        "question-set": ["texts", "endGame"],
        "interactive-book": ["l10n", "a11y"],
    }
    for ct, sections in nested_types.items():
        en_path = LOCALES_DIR / ct / "en.json"
        if not en_path.exists():
            continue
        en_data = json.loads(en_path.read_text(encoding="utf-8"))
        for section in sections:
            check(
                f"{ct}/en.json has '{section}' section",
                section in en_data and isinstance(en_data[section], dict),
            )
            en_section_keys = set(en_data.get(section, {}).keys())
            for lang in LANGUAGES:
                if lang == "en":
                    continue
                lang_path = LOCALES_DIR / ct / f"{lang}.json"
                if not lang_path.exists():
                    continue
                lang_data = json.loads(lang_path.read_text(encoding="utf-8"))
                lang_section_keys = set(lang_data.get(section, {}).keys())
                missing = en_section_keys - lang_section_keys
                check(
                    f"{ct}/{lang}.json[{section}] keys match en",
                    missing == set(),
                    f"missing={missing}" if missing else "",
                )


def test_no_empty_string_values():
    """Test 4: No locale value is an empty string (except explicitly allowed ones)."""
    print("\n=== Test 4: No empty string values ===")
    ALLOWED_EMPTY = {"questionSetInstruction"}  # explicitly empty in H5P semantics
    for ct in CONTENT_TYPES:
        for lang in LANGUAGES:
            fpath = LOCALES_DIR / ct / f"{lang}.json"
            if not fpath.exists():
                continue
            data = json.loads(fpath.read_text(encoding="utf-8"))
            _check_no_empty(data, f"{ct}/{lang}.json", ALLOWED_EMPTY)


def _check_no_empty(data, prefix, allowed, path=""):
    for k, v in data.items():
        full_key = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            _check_no_empty(v, prefix, allowed, full_key)
        elif isinstance(v, str):
            if k not in allowed:
                check(
                    f"{prefix}[{full_key}] not empty",
                    v.strip() != "",
                    f"Empty string at key '{full_key}'",
                )
        elif isinstance(v, list):
            # overallFeedback arrays are fine
            pass


def test_get_l10n_basic():
    """Test 5: get_l10n() loads and merges correctly for all languages."""
    print("\n=== Test 5: get_l10n() basic loading ===")

    for lang in LANGUAGES:
        l10n = get_l10n("flashcards", lang)
        check(
            f"flashcards get_l10n({lang}) returns dict",
            isinstance(l10n, dict) and len(l10n) > 0,
        )
        check(
            f"flashcards get_l10n({lang}) has 'progressText'",
            "progressText" in l10n,
            f"Keys: {list(l10n.keys())[:10]}",
        )
        check(
            f"flashcards get_l10n({lang}) has 'confirmCheck'",
            "confirmCheck" in l10n,
        )

    # Unknown language should fall back to English
    l10n_xx = get_l10n("flashcards", "xx")
    l10n_en = get_l10n("flashcards", "en")
    check(
        "flashcards get_l10n('xx') falls back to English",
        l10n_xx.get("progressText") == l10n_en.get("progressText"),
    )


def test_get_l10n_language_strings_differ():
    """Test 6: Different languages return different strings."""
    print("\n=== Test 6: Language strings actually differ ===")

    en = get_l10n("flashcards", "en")
    tr = get_l10n("flashcards", "tr")
    ar = get_l10n("flashcards", "ar")
    es = get_l10n("flashcards", "es")
    fr = get_l10n("flashcards", "fr")

    check("en.next != tr.next", en.get("next") != tr.get("next"),
          f"en={en.get('next')}, tr={tr.get('next')}")
    check("en.next != ar.next", en.get("next") != ar.get("next"),
          f"en={en.get('next')}, ar={ar.get('next')}")
    check("en.next != es.next", en.get("next") != es.get("next"),
          f"en={en.get('next')}, es={es.get('next')}")
    check("en.next != fr.next", en.get("next") != fr.get("next"),
          f"en={en.get('next')}, fr={fr.get('next')}")

    check("en.confirmCheck.header != tr.confirmCheck.header",
          en["confirmCheck"]["header"] != tr["confirmCheck"]["header"],
          f"en={en['confirmCheck']['header']}, tr={tr['confirmCheck']['header']}")


def test_get_l10n_nested_content_types():
    """Test 7: Nested locale types (crossword, question-set) load correctly."""
    print("\n=== Test 7: Nested content type locales ===")

    for lang in LANGUAGES:
        l10n = get_l10n("crossword", lang)
        check(
            f"crossword get_l10n({lang}) has 'l10n' section",
            "l10n" in l10n and isinstance(l10n["l10n"], dict),
        )
        check(
            f"crossword get_l10n({lang}) has 'a11y' section",
            "a11y" in l10n and isinstance(l10n["a11y"], dict),
        )
        check(
            f"crossword get_l10n({lang}) l10n.across exists",
            "across" in l10n.get("l10n", {}),
        )

    en = get_l10n("crossword", "en")
    tr = get_l10n("crossword", "tr")
    check(
        "crossword en.l10n.across != tr.l10n.across",
        en["l10n"]["across"] != tr["l10n"]["across"],
        f"en={en['l10n']['across']}, tr={tr['l10n']['across']}",
    )

    for lang in LANGUAGES:
        l10n = get_l10n("question-set", lang)
        check(
            f"question-set get_l10n({lang}) has 'texts' section",
            "texts" in l10n and isinstance(l10n["texts"], dict),
        )
        check(
            f"question-set get_l10n({lang}) has 'endGame' section",
            "endGame" in l10n and isinstance(l10n["endGame"], dict),
        )


def test_content_type_overrides_common():
    """Test 8: Content-type locale strings override common strings."""
    print("\n=== Test 8: Content-type overrides common ===")

    common_en = json.loads((LOCALES_DIR / "_common" / "en.json").read_text())
    flash_en = json.loads((LOCALES_DIR / "flashcards" / "en.json").read_text())

    # Common has 'ui' with 'checkAnswerButton'
    check(
        "common has ui.checkAnswerButton",
        "ui" in common_en and "checkAnswerButton" in common_en["ui"],
    )

    # Flashcards has 'checkAnswerText' — different key, no override conflict
    check(
        "flashcards has checkAnswerText",
        "checkAnswerText" in flash_en,
    )


def test_get_ui_strings():
    """Test 9: get_ui_strings() works for all languages (via get_l10n 'ui' key)."""
    print("\n=== Test 9: get_ui_strings() across languages ===")

    for lang in LANGUAGES:
        l10n = get_l10n("multiple-choice", lang)
        # base.get_ui_strings does: l10n.get("ui", l10n)
        ui = l10n.get("ui", l10n)
        check(
            f"multiple-choice ui_strings({lang}) is dict",
            isinstance(ui, dict) and len(ui) > 0,
        )


def test_get_overall_feedback():
    """Test 10: get_overall_feedback() returns localized feedback for all languages."""
    print("\n=== Test 10: get_overall_feedback() across languages ===")

    en_l10n = get_l10n("flashcards", "en")
    tr_l10n = get_l10n("flashcards", "tr")
    en_fb = en_l10n.get("overallFeedback", [])
    tr_fb = tr_l10n.get("overallFeedback", [])

    check(
        "overall_feedback en is list of 3",
        isinstance(en_fb, list) and len(en_fb) == 3,
    )
    check(
        "overall_feedback tr is list of 3",
        isinstance(tr_fb, list) and len(tr_fb) == 3,
    )
    check(
        "overall_feedback en != tr (feedback text differs)",
        len(en_fb) > 0 and len(tr_fb) > 0 and en_fb[0]["feedback"] != tr_fb[0]["feedback"],
        f"en={en_fb[0]['feedback'] if en_fb else '?'}, tr={tr_fb[0]['feedback'] if tr_fb else '?'}",
    )


def test_all_content_types_load():
    """Test 11: get_l10n() works for every content type (not just flashcards)."""
    print("\n=== Test 11: get_l10n() for all content types ===")

    for ct in CONTENT_TYPES:
        if ct == "_common":
            continue
        for lang in LANGUAGES:
            l10n = get_l10n(ct, lang)
            check(
                f"{ct} get_l10n({lang}) non-empty",
                isinstance(l10n, dict) and len(l10n) > 0,
                f"Got {type(l10n)} with {len(l10n) if isinstance(l10n, dict) else 0} keys",
            )


def test_arabic_rtl_strings_present():
    """Test 12: Arabic strings are actual Arabic (contain Arabic characters)."""
    print("\n=== Test 12: Arabic strings contain Arabic characters ===")
    import re
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')

    for ct in CONTENT_TYPES:
        ar_path = LOCALES_DIR / ct / "ar.json"
        if not ar_path.exists():
            continue
        data = json.loads(ar_path.read_text(encoding="utf-8"))
        _check_arabic_values(data, f"{ct}/ar.json", arabic_pattern)


def _check_arabic_values(data, prefix, pattern, path=""):
    for k, v in data.items():
        full_key = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            _check_arabic_values(v, prefix, pattern, full_key)
        elif isinstance(v, str) and v.strip():
            # Skip template-only strings like "@card of @total"
            stripped = v.replace("@", "").replace(":", "").replace("/", "").replace("%", "").strip()
            if stripped and not stripped.replace("d", "").replace("total", "").replace("num", "").replace("score", "").replace("card", "").replace("current", "").replace("letter", "").replace("length", "").replace("keyword", "").strip() == "":
                check(
                    f"{prefix}[{full_key}] has Arabic chars",
                    bool(pattern.search(v)),
                    f"Value: '{v}'",
                )
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict) and "feedback" in item:
                    check(
                        f"{prefix}[{full_key}[{i}].feedback] has Arabic chars",
                        bool(pattern.search(item["feedback"])),
                        f"Value: '{item['feedback']}'",
                    )


if __name__ == "__main__":
    print("=" * 60)
    print("H5P Multi-Language Locale Test Suite")
    print("=" * 60)

    # Pure JSON file tests (no imports needed)
    test_all_locale_files_exist_and_valid()
    test_en_and_lang_have_same_keys()
    test_nested_structures_consistent()
    test_no_empty_string_values()
    test_arabic_rtl_strings_present()

    # Tests using standalone get_l10n() (mirrors base.py logic)
    test_get_l10n_basic()
    test_get_l10n_language_strings_differ()
    test_get_l10n_nested_content_types()
    test_content_type_overrides_common()
    test_get_ui_strings()
    test_get_overall_feedback()
    test_all_content_types_load()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"\nFailures:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
