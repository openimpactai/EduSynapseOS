# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
#!/usr/bin/env python3
"""
Content Creation - Scenario-Based QC Tests
============================================

Tests realistic content creation scenarios for different user types:
- UK Teacher creating Maths quiz
- USA Student practicing Science
- Rwanda Parent helping with homework
- Malawi Teacher creating differentiated content
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

# Load playground data
PLAYGROUND_FILE = "/home/erhanarslan/work/gdlabs/projects/edusynapse/scripts/.playground-result.json"

with open(PLAYGROUND_FILE) as f:
    PLAYGROUND = json.load(f)


class ScenarioResult:
    def __init__(self, name: str):
        self.name = name
        self.steps = []
        self.success = True

    def add_step(self, step: str, passed: bool, details: str = ""):
        self.steps.append({
            "step": step,
            "passed": passed,
            "details": details
        })
        if not passed:
            self.success = False


def print_scenario_header(title: str, user_type: str, country: str):
    """Print a scenario header."""
    console.print(Panel.fit(
        f"[bold cyan]{title}[/bold cyan]\n"
        f"User Type: [yellow]{user_type}[/yellow] | Country: [green]{country}[/green]",
        border_style="blue"
    ))


async def scenario_uk_teacher_quiz():
    """
    Scenario: UK Teacher Emma Wilson creates a Maths quiz

    Use Case: UC-T1 - Create quiz for specific topic
    Framework: UK-NC-2014
    Subject: Mathematics
    Topic: Fractions
    Grade: Year 6
    """
    print_scenario_header(
        "UK Teacher Creates Maths Quiz",
        "Teacher",
        "United Kingdom"
    )

    result = ScenarioResult("UK Teacher Quiz Creation")
    teacher = PLAYGROUND["teachers"]["uk"]

    console.print(f"\nTeacher: [cyan]{teacher['name']}[/cyan]")
    console.print(f"Email: {teacher['email']}")
    console.print(f"Framework: UK-NC-2014")
    console.print("")

    # Step 1: Verify teacher data
    try:
        assert teacher["id"], "Teacher ID missing"
        assert teacher["email"], "Teacher email missing"
        result.add_step("Verify teacher profile", True, f"ID: {teacher['id'][:8]}...")
        console.print("  [green]✓[/green] Teacher profile verified")
    except AssertionError as e:
        result.add_step("Verify teacher profile", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 2: Simulate content request
    request_data = {
        "message": "Create a 5-question quiz about fractions for Year 6",
        "context": {
            "user_role": "teacher",
            "framework_code": "UK-NC-2014",
            "subject_code": "MATH",
            "topic_code": "FRACTIONS",
            "grade_level": 6,
            "country_code": "GB",
            "language": "en"
        }
    }
    result.add_step("Prepare content request", True, json.dumps(request_data, indent=2)[:100])
    console.print("  [green]✓[/green] Content request prepared")

    # Step 3: Test converter availability
    try:
        from src.services.h5p.converters.registry import ConverterRegistry
        registry = ConverterRegistry()

        # Check multiple choice is available
        mc_converter = registry.get("multiple-choice")
        assert mc_converter, "Multiple choice converter not found"
        result.add_step("Multiple choice converter ready", True)
        console.print("  [green]✓[/green] Multiple choice converter available")

        # Check flashcards for vocabulary support
        fc_converter = registry.get("flashcards")
        assert fc_converter, "Flashcards converter not found"
        result.add_step("Flashcards converter ready", True)
        console.print("  [green]✓[/green] Flashcards converter available")

    except Exception as e:
        result.add_step("Converter check", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 4: Test conversion with sample content
    try:
        sample_quiz = {
            "questions": [
                {
                    "question": "What is 1/2 + 1/4?",
                    "answers": ["3/4", "2/6", "1/6", "2/4"],
                    "correctIndex": 0,
                    "explanation": "To add fractions, find a common denominator. 1/2 = 2/4, so 2/4 + 1/4 = 3/4"
                },
                {
                    "question": "Simplify the fraction 4/8",
                    "answers": ["1/2", "2/4", "4/8", "1/4"],
                    "correctIndex": 0,
                    "explanation": "4/8 simplifies to 1/2 by dividing both by 4"
                }
            ]
        }

        h5p_params = mc_converter.convert(sample_quiz, language="en")
        assert "question" in h5p_params, "Missing question in output"
        assert "answers" in h5p_params, "Missing answers in output"

        result.add_step("Quiz conversion successful", True)
        console.print("  [green]✓[/green] Sample quiz converted to H5P format")

    except Exception as e:
        result.add_step("Quiz conversion", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 5: Check agent configuration
    try:
        import yaml
        config_path = "/home/erhanarslan/work/gdlabs/projects/edusynapse/edusynapseos-backend/config/agents/content/quiz_generator.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["agent"]["id"] == "quiz_generator", "Invalid agent ID"
        result.add_step("Quiz generator agent configured", True)
        console.print("  [green]✓[/green] Quiz generator agent configuration valid")

    except Exception as e:
        result.add_step("Agent config check", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    return result


async def scenario_usa_student_practice():
    """
    Scenario: USA Student Emily Davis practices Science

    Use Case: UC-S1 - Practice specific topic on-demand
    Framework: CCSS
    Subject: Science
    Topic: Photosynthesis
    Grade: 7
    """
    print_scenario_header(
        "USA Student Practices Science",
        "Student",
        "United States"
    )

    result = ScenarioResult("USA Student Practice")
    student = PLAYGROUND["students"]["usa"]

    console.print(f"\nStudent: [cyan]{student['name']}[/cyan]")
    console.print(f"Email: {student['email']}")
    console.print(f"Framework: CCSS")
    console.print("")

    # Step 1: Verify student data
    try:
        assert student["id"], "Student ID missing"
        assert student["framework_code"] == "CCSS", "Wrong framework"
        result.add_step("Verify student profile", True)
        console.print("  [green]✓[/green] Student profile verified")
    except AssertionError as e:
        result.add_step("Verify student profile", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 2: Simulate practice request
    request_data = {
        "message": "I want to practice photosynthesis - can you give me some flashcards?",
        "context": {
            "user_role": "student",
            "framework_code": "CCSS",
            "subject_code": "SCI",
            "topic_code": "PHOTOSYNTHESIS",
            "grade_level": 7,
            "country_code": "US",
            "language": "en"
        }
    }
    result.add_step("Prepare practice request", True)
    console.print("  [green]✓[/green] Practice request prepared")

    # Step 3: Test flashcard generation
    try:
        from src.services.h5p.converters.flashcards import FlashcardsConverter
        converter = FlashcardsConverter()

        sample_content = {
            "cards": [
                {
                    "term": "Photosynthesis",
                    "definition": "Process by which plants convert sunlight into chemical energy"
                },
                {
                    "term": "Chlorophyll",
                    "definition": "Green pigment that absorbs light energy in plants"
                },
                {
                    "term": "Glucose",
                    "definition": "Sugar produced as a result of photosynthesis"
                },
                {
                    "term": "Carbon Dioxide",
                    "definition": "Gas absorbed by plants from the air during photosynthesis"
                }
            ],
            "description": "Key terms for photosynthesis"
        }

        h5p_params = converter.convert(sample_content, language="en")
        assert len(h5p_params["cards"]) == 4, "Should have 4 cards"

        result.add_step("Flashcard generation successful", True)
        console.print("  [green]✓[/green] Flashcards generated (4 cards)")

    except Exception as e:
        result.add_step("Flashcard generation", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 4: Test drag words for practice
    try:
        from src.services.h5p.converters.drag_words import DragWordsConverter
        converter = DragWordsConverter()

        sample_content = {
            "exercises": [
                {
                    "instruction": "Complete the photosynthesis equation",
                    "text": "*Sunlight* + *water* + carbon dioxide → glucose + *oxygen*",
                    "draggables": ["Sunlight", "water", "oxygen"]
                }
            ]
        }

        h5p_params = converter.convert(sample_content, language="en")
        assert "textField" in h5p_params, "Missing text field"

        result.add_step("Drag words exercise ready", True)
        console.print("  [green]✓[/green] Drag words exercise available")

    except Exception as e:
        result.add_step("Drag words", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    return result


async def scenario_rwanda_parent_homework():
    """
    Scenario: Rwanda Parent Claude Mugisha helps child with homework

    Use Case: UC-P1 - Homework support materials
    Framework: RW-CBC
    Child: Diane Mugisha
    Subject: Mathematics
    Language: English (Kinyarwanda framework, English medium)
    """
    print_scenario_header(
        "Rwanda Parent Supports Homework",
        "Parent",
        "Rwanda"
    )

    result = ScenarioResult("Rwanda Parent Homework Support")
    parent = PLAYGROUND["parents"]["rwanda"]
    student = PLAYGROUND["students"]["rwanda"]

    console.print(f"\nParent: [cyan]{parent['name']}[/cyan]")
    console.print(f"Child: [yellow]{parent['child']}[/yellow]")
    console.print(f"Framework: RW-CBC (Competence-Based Curriculum)")
    console.print("")

    # Step 1: Verify parent-child relationship
    try:
        assert parent["child"] == student["name"], "Parent-child mismatch"
        result.add_step("Verify parent-child relationship", True)
        console.print("  [green]✓[/green] Parent-child relationship verified")
    except AssertionError as e:
        result.add_step("Verify relationship", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 2: Check framework support
    try:
        assert student["framework_code"] == "RW-CBC", "Wrong framework"
        result.add_step("RW-CBC framework configured", True)
        console.print("  [green]✓[/green] Rwanda CBC framework supported")
    except AssertionError as e:
        result.add_step("Framework check", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 3: Generate simple practice materials
    try:
        from src.services.h5p.converters.fill_blanks import FillBlanksConverter
        converter = FillBlanksConverter()

        # Simple homework-style content
        sample_content = {
            "exercises": [
                {
                    "text": "In Rwanda, the capital city is *Kigali*.",
                    "blanks": ["Kigali"]
                },
                {
                    "text": "Lake *Kivu* is one of the great lakes in Rwanda.",
                    "blanks": ["Kivu"]
                }
            ]
        }

        h5p_params = converter.convert(sample_content, language="en")
        assert "text" in h5p_params, "Missing text field"

        result.add_step("Homework materials generated", True)
        console.print("  [green]✓[/green] Homework support materials ready")

    except Exception as e:
        result.add_step("Homework materials", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    return result


async def scenario_malawi_teacher_differentiated():
    """
    Scenario: Malawi Teacher Grace Banda creates differentiated content

    Use Case: UC-T4 - Create differentiated content for different levels
    Framework: MW-NC (Malawi National Curriculum)
    Subject: English
    Different levels: Struggling, On-level, Advanced
    """
    print_scenario_header(
        "Malawi Teacher Creates Differentiated Content",
        "Teacher",
        "Malawi"
    )

    result = ScenarioResult("Malawi Teacher Differentiated Content")
    teacher = PLAYGROUND["teachers"]["malawi"]

    console.print(f"\nTeacher: [cyan]{teacher['name']}[/cyan]")
    console.print(f"School: Malawi School ID: {PLAYGROUND['schools']['malawi'][:8]}...")
    console.print(f"Framework: MW-NC")
    console.print("")

    # Step 1: Verify teacher data
    try:
        assert teacher["id"], "Teacher ID missing"
        assert teacher["country_code"] == "MW", "Wrong country code"
        result.add_step("Verify teacher profile", True)
        console.print("  [green]✓[/green] Teacher profile verified")
    except AssertionError as e:
        result.add_step("Verify teacher", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 2: Test easy level content (struggling learners)
    try:
        from src.services.h5p.converters.true_false import TrueFalseConverter
        converter = TrueFalseConverter()

        easy_content = {
            "statements": [
                {
                    "statement": "A noun is a naming word.",
                    "isTrue": True,
                    "explanation": "Correct! Nouns name people, places, things, or ideas."
                }
            ]
        }

        h5p_params = converter.convert(easy_content, language="en")
        assert "correct" in h5p_params, "Missing correct field"

        result.add_step("Easy level content (True/False)", True)
        console.print("  [green]✓[/green] Easy level: True/False questions ready")

    except Exception as e:
        result.add_step("Easy level", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 3: Test medium level content (on-level learners)
    try:
        from src.services.h5p.converters.mark_words import MarkWordsConverter
        converter = MarkWordsConverter()

        medium_content = {
            "exercises": [
                {
                    "instruction": "Click on all the nouns in this sentence.",
                    "text": "The *cat* sat on the *mat* in the *house*.",
                    "targetWords": ["cat", "mat", "house"],
                    "wordType": "nouns"
                }
            ]
        }

        h5p_params = converter.convert(medium_content, language="en")
        assert "textField" in h5p_params, "Missing text field"

        result.add_step("Medium level content (Mark Words)", True)
        console.print("  [green]✓[/green] Medium level: Mark the Words ready")

    except Exception as e:
        result.add_step("Medium level", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 4: Test advanced level content (Essay)
    try:
        from src.services.h5p.converters.essay import EssayConverter
        converter = EssayConverter()

        advanced_content = {
            "prompt": "Write a paragraph using at least 5 different nouns. Underline each noun.",
            "sampleOutline": "Include: person, place, thing, animal, idea",
            "keywords": [
                {"keyword": "noun", "alternatives": ["nouns"], "points": 2}
            ],
            "minimumLength": 50
        }

        h5p_params = converter.convert(advanced_content, language="en")
        assert "taskDescription" in h5p_params, "Missing task description"

        result.add_step("Advanced level content (Essay)", True)
        console.print("  [green]✓[/green] Advanced level: Essay prompt ready")

    except Exception as e:
        result.add_step("Advanced level", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    # Step 5: Verify differentiation support in agent config
    try:
        import yaml
        config_path = "/home/erhanarslan/work/gdlabs/projects/edusynapse/edusynapseos-backend/config/agents/content/learning_content_generator.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "differentiation_level" in str(config), "Missing differentiation support"
        result.add_step("Differentiation support configured", True)
        console.print("  [green]✓[/green] Differentiation levels supported in agent")

    except Exception as e:
        result.add_step("Differentiation config", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    return result


async def scenario_translation_test():
    """
    Scenario: Test multi-language support (Turkish, English)

    Use Case: UC-T6 - Translate content to different languages
    """
    print_scenario_header(
        "Multi-Language Content Translation",
        "System",
        "Global"
    )

    result = ScenarioResult("Translation Support")

    # Test Turkish localization
    try:
        from src.services.h5p.converters.multiple_choice import MultipleChoiceConverter
        converter = MultipleChoiceConverter()

        content = {
            "questions": [{
                "question": "2 + 2 = ?",
                "answers": ["4", "3", "5", "6"],
                "correctIndex": 0
            }]
        }

        # English version
        en_params = converter.convert(content, language="en")
        # Turkish version
        tr_params = converter.convert(content, language="tr")

        assert en_params["UI"]["checkAnswerButton"] == "Check", "English UI missing"
        assert tr_params["UI"]["checkAnswerButton"] == "Kontrol Et", "Turkish UI missing"

        result.add_step("English localization", True)
        console.print("  [green]✓[/green] English UI strings correct")
        result.add_step("Turkish localization", True)
        console.print("  [green]✓[/green] Turkish UI strings correct")

    except Exception as e:
        result.add_step("Localization test", False, str(e))
        console.print(f"  [red]✗[/red] {e}")

    return result


async def run_all_scenarios():
    """Run all scenario tests."""
    console.print(Panel.fit(
        "[bold magenta]Content Creation Scenario Tests[/bold magenta]\n"
        "[dim]Testing realistic user scenarios with playground data[/dim]\n"
        f"[dim]Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="magenta"
    ))

    scenarios = [
        scenario_uk_teacher_quiz,
        scenario_usa_student_practice,
        scenario_rwanda_parent_homework,
        scenario_malawi_teacher_differentiated,
        scenario_translation_test,
    ]

    results = []
    for scenario_func in scenarios:
        console.print("\n" + "=" * 60 + "\n")
        result = await scenario_func()
        results.append(result)
        console.print("")

    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel.fit("[bold]Scenario Test Summary[/bold]", border_style="cyan"))

    table = Table(title="Scenario Results")
    table.add_column("Scenario", style="cyan")
    table.add_column("Steps", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Status")

    total_steps = 0
    total_passed = 0
    total_failed = 0

    for r in results:
        passed = sum(1 for s in r.steps if s["passed"])
        failed = len(r.steps) - passed
        total_steps += len(r.steps)
        total_passed += passed
        total_failed += failed

        status = "[green]✓ PASS[/green]" if r.success else "[red]✗ FAIL[/red]"
        table.add_row(r.name, str(len(r.steps)), str(passed), str(failed), status)

    console.print(table)

    console.print(f"\n[bold]Total Steps: {total_steps}[/bold] | "
                 f"[green]Passed: {total_passed}[/green] | "
                 f"[red]Failed: {total_failed}[/red]")

    success_rate = (total_passed / total_steps * 100) if total_steps > 0 else 0
    console.print(f"\n[bold]Overall Success Rate: {success_rate:.1f}%[/bold]")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": [
            {
                "name": r.name,
                "success": r.success,
                "steps": r.steps
            }
            for r in results
        ],
        "summary": {
            "total_steps": total_steps,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": success_rate
        }
    }

    output_path = "/home/erhanarslan/work/gdlabs/projects/edusynapse/edusynapseos-backend/tests/qc_scenarios_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(f"\n[dim]Results saved to: {output_path}[/dim]")

    return total_failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_scenarios())
    sys.exit(0 if success else 1)
