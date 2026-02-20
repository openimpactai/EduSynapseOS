# H5P Content Type Schemas

This directory contains JSON schema examples for all supported H5P content types.
These schemas are used by AI agents to generate properly formatted H5P content.

## Directory Structure

```
h5p-schemas/
├── README.md                       # This file
├── content-types.json              # Master list of all content types (30 types)
├── assessment/                     # Quiz and assessment types (10 schemas)
│   ├── multiple-choice.json        # Multiple choice questions
│   ├── true-false.json             # True/false statements
│   ├── fill-blanks.json            # Fill in the blanks
│   ├── drag-words.json             # Drag words into text
│   ├── mark-the-words.json         # Click to mark words
│   ├── single-choice-set.json      # Rapid-fire questions
│   ├── essay.json                  # Open-ended writing
│   ├── summary.json                # Statement selection
│   ├── question-set.json           # Composite quiz
│   └── arithmetic-quiz.json        # Math calculations
├── vocabulary/                     # Vocabulary and memorization (4 schemas)
│   ├── flashcards.json             # Flip cards
│   ├── dialog-cards.json           # Dialog-style cards
│   ├── crossword.json              # Crossword puzzles
│   └── find-the-words.json         # Word search
├── learning/                       # Learning and teaching content (7 schemas)
│   ├── course-presentation.json    # Interactive slideshow
│   ├── interactive-book.json       # Multi-chapter book
│   ├── accordion.json              # Expandable panels
│   ├── timeline.json               # Chronological timeline
│   ├── image-hotspots.json         # Interactive image points
│   ├── branching-scenario.json     # Decision-based learning
│   └── documentation-tool.json     # Structured documentation
├── game/                           # Game-based content (5 schemas)
│   ├── memory-game.json            # Card matching
│   ├── personality-quiz.json       # Personality assessment
│   ├── sort-paragraphs.json        # Order paragraphs
│   ├── image-sequencing.json       # Order images
│   └── image-pairing.json          # Match image pairs
└── media/                          # Media-based content (4 schemas)
    ├── chart.json                  # Bar/pie charts
    ├── image-juxtaposition.json    # Before/after slider
    ├── agamotto.json               # Image sequence slider
    └── collage.json                # Image collage
```

## Content Type Categories

### Assessment & Quiz (10 types)
- **Full AI Support**: Multiple Choice, True/False, Fill Blanks, Drag Words, Mark Words, Single Choice Set, Essay, Summary, Question Set, Arithmetic Quiz

### Vocabulary & Memorization (4 types)
- **Full AI Support**: Flashcards, Dialog Cards, Crossword, Word Search

### Learning & Teaching (7 types)
- **Full AI Support**: Accordion, Documentation Tool
- **Partial AI Support** (requires images): Course Presentation, Interactive Book, Timeline, Image Hotspots, Branching Scenario

### Game-Based Learning (5 types)
- **Full AI Support**: Personality Quiz, Sort Paragraphs
- **Partial AI Support** (requires images): Memory Game, Image Sequencing, Image Pairing

### Media & Visualization (4 types)
- **Full AI Support**: Chart
- **Partial AI Support** (requires images): Image Juxtaposition, Agamotto, Collage

## AI Support Levels

| Level | Description |
|-------|-------------|
| **Full** | AI can generate complete content without additional resources |
| **Partial** | AI generates structure/text, but requires images to be provided or generated |

## Schema Format

Each schema file contains:
- `library`: H5P library identifier (e.g., "H5P.MultiChoice 1.16")
- `name`: Human-readable name
- `category`: Content category
- `description`: What this content type is for
- `ai_input_format`: Expected format from AI generation with examples
- `h5p_params_format`: Exact H5P params structure with examples
- `conversion_notes`: How to convert from AI format to H5P format
- `requires_media`: Whether the content type needs images/media
- `ai_support`: Level of AI support ("full" or "partial")

## Usage

### Loading a Schema
```python
# Example: Loading schema for Multiple Choice
schema = load_h5p_schema("assessment/multiple-choice")

# Access AI input format
ai_format = schema["ai_input_format"]["schema"]

# Access H5P params format
h5p_format = schema["h5p_params_format"]
```

### Generating Content
```python
# 1. Generate AI content based on ai_input_format
ai_content = ai_generate(topic, schema["ai_input_format"])

# 2. Convert to H5P params using conversion_notes
h5p_params = convert_to_h5p(ai_content, schema)

# 3. Create H5P content
h5p_content = {
    "library": schema["library"],
    "params": h5p_params
}
```

## Total: 30 Content Types

| Category | Count | Full AI | Partial AI |
|----------|-------|---------|------------|
| Assessment | 10 | 10 | 0 |
| Vocabulary | 4 | 4 | 0 |
| Learning | 7 | 2 | 5 |
| Game | 5 | 2 | 3 |
| Media | 4 | 1 | 3 |
| **Total** | **30** | **19** | **11** |
