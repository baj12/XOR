"""
Substance Vocabulary System

Provides controlled vocabulary for substances used in continuous learning experiments.
Maps substance names to class labels with case-insensitive matching and misspelling handling.

Author: Claude Code
Date: 2026-01-04
"""

from typing import Dict, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# Canonical substance-to-class mapping
# Class 0: Control/empty substances
# Class 1+: Positive substances (lavender, peppermint, etc.)
SUBSTANCE_VOCABULARY: Dict[str, int] = {
    # Negative class (0) - controls and empty
    'empty': 0,
    'control': 0,
    'water': 0,
    'air': 0,
    'blank': 0,
    'nothing': 0,

    # Positive classes (1+)
    'lavender': 1,
    'lavendar': 1,      # common misspelling
    'lavander': 1,      # another misspelling
    'peppermint': 2,
    'eucalyptus': 3,
    'rosemary': 4,
    'tea_tree': 5,
    'lemon': 6,
    'orange': 7,
}


# Reverse mapping: class -> canonical substance names
CLASS_TO_SUBSTANCE: Dict[int, Set[str]] = {}
for substance, cls in SUBSTANCE_VOCABULARY.items():
    if cls not in CLASS_TO_SUBSTANCE:
        CLASS_TO_SUBSTANCE[cls] = set()
    CLASS_TO_SUBSTANCE[cls].add(substance)


def normalize_substance_name(substance: str) -> str:
    """
    Normalize substance name for lookup.

    Args:
        substance: Raw substance name (e.g., "Lavender", "EMPTY", "tea tree")

    Returns:
        Normalized name (lowercase, underscores for spaces)
    """
    return substance.lower().strip().replace(' ', '_')


def get_class_for_substance(substance: str) -> Optional[int]:
    """
    Get class label for a substance name.

    Args:
        substance: Substance name (case-insensitive, spaces allowed)

    Returns:
        Class label (int) or None if not found

    Examples:
        >>> get_class_for_substance("Lavender")
        1
        >>> get_class_for_substance("EMPTY")
        0
        >>> get_class_for_substance("tea tree")
        5
        >>> get_class_for_substance("unknown")
        None
    """
    normalized = normalize_substance_name(substance)
    return SUBSTANCE_VOCABULARY.get(normalized)


def validate_substance(substance: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a substance name and provide helpful error messages.

    Args:
        substance: Substance name to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, None)
        If invalid: (False, "error message with suggestions")

    Examples:
        >>> validate_substance("lavender")
        (True, None)
        >>> validate_substance("unknown")
        (False, "Unknown substance 'unknown'. Valid options: ...")
    """
    normalized = normalize_substance_name(substance)

    if normalized in SUBSTANCE_VOCABULARY:
        return True, None

    # Provide helpful suggestions
    all_substances = sorted(set(SUBSTANCE_VOCABULARY.keys()))

    # Check for close matches
    suggestions = []
    for valid_name in all_substances:
        if normalized in valid_name or valid_name in normalized:
            suggestions.append(valid_name)

    if suggestions:
        error_msg = (f"Unknown substance '{substance}'. "
                    f"Did you mean one of: {', '.join(suggestions)}?")
    else:
        error_msg = (f"Unknown substance '{substance}'. "
                    f"Valid options: {', '.join(all_substances)}")

    return False, error_msg


def get_canonical_name(substance: str) -> Optional[str]:
    """
    Get the first canonical name for a substance class.

    Useful for display purposes when you want to show the "primary" name
    for a class, even if the user entered a misspelling or alias.

    Args:
        substance: Substance name (can be alias/misspelling)

    Returns:
        Canonical name (first in alphabetical order for that class) or None

    Examples:
        >>> get_canonical_name("lavendar")  # misspelling
        'lavander'  # returns first alphabetically for class 1
        >>> get_canonical_name("empty")
        'air'  # returns first alphabetically for class 0
    """
    cls = get_class_for_substance(substance)
    if cls is None:
        return None

    # Return first canonical name for this class (alphabetically)
    canonical_names = sorted(CLASS_TO_SUBSTANCE.get(cls, []))
    return canonical_names[0] if canonical_names else None


def get_all_substances_for_class(cls: int) -> Set[str]:
    """
    Get all substance names (including aliases) for a class.

    Args:
        cls: Class label

    Returns:
        Set of substance names

    Examples:
        >>> get_all_substances_for_class(1)
        {'lavender', 'lavendar', 'lavander'}
    """
    return CLASS_TO_SUBSTANCE.get(cls, set())


def get_all_classes() -> Set[int]:
    """Get all valid class labels."""
    return set(SUBSTANCE_VOCABULARY.values())


def get_substance_choices() -> list:
    """
    Get unique substance choices for UI dropdowns.

    Returns canonical names only (no aliases/misspellings).
    Sorted by class, then alphabetically within class.

    Returns:
        List of (substance_name, class_label, display_name) tuples

    Example:
        [('air', 0, 'Air (Class 0)'),
         ('blank', 0, 'Blank (Class 0)'),
         ('lavender', 1, 'Lavender (Class 1)'),
         ...]
    """
    choices = []

    # Get unique substances (prefer shortest name for each meaning)
    seen_meanings = {}
    for substance, cls in SUBSTANCE_VOCABULARY.items():
        key = (cls, substance.replace('_', ' '))
        if key not in seen_meanings or len(substance) < len(seen_meanings[key]):
            seen_meanings[key] = substance

    # Create choices
    for (cls, display), substance in seen_meanings.items():
        display_name = display.title().replace('_', ' ')
        choices.append((substance, cls, f"{display_name} (Class {cls})"))

    # Sort by class, then alphabetically
    choices.sort(key=lambda x: (x[1], x[2]))

    return choices


def add_substance(name: str, class_label: int) -> bool:
    """
    Add a new substance to the vocabulary (runtime extension).

    Args:
        name: Substance name (will be normalized)
        class_label: Class label to assign

    Returns:
        True if added, False if already exists

    Note:
        This modifies the runtime vocabulary only. To persist changes,
        update the SUBSTANCE_VOCABULARY dictionary in this file.
    """
    normalized = normalize_substance_name(name)

    if normalized in SUBSTANCE_VOCABULARY:
        logger.warning(f"Substance '{name}' already exists with class {SUBSTANCE_VOCABULARY[normalized]}")
        return False

    SUBSTANCE_VOCABULARY[normalized] = class_label

    if class_label not in CLASS_TO_SUBSTANCE:
        CLASS_TO_SUBSTANCE[class_label] = set()
    CLASS_TO_SUBSTANCE[class_label].add(normalized)

    logger.info(f"Added substance '{name}' -> class {class_label}")
    return True


if __name__ == '__main__':
    # Test the vocabulary system
    print("Substance Vocabulary System - Test")
    print("=" * 50)

    # Test lookups
    test_cases = [
        "Lavender",
        "EMPTY",
        "tea tree",
        "lavendar",  # misspelling
        "unknown",
    ]

    print("\n1. Testing substance lookups:")
    for substance in test_cases:
        cls = get_class_for_substance(substance)
        print(f"  '{substance}' -> class {cls}")

    # Test validation
    print("\n2. Testing validation:")
    for substance in ["lavender", "unknown", "laven"]:
        is_valid, error = validate_substance(substance)
        if is_valid:
            print(f"  '{substance}': VALID")
        else:
            print(f"  '{substance}': INVALID - {error}")

    # Test canonical names
    print("\n3. Testing canonical names:")
    for substance in ["lavendar", "empty", "tea tree"]:
        canonical = get_canonical_name(substance)
        print(f"  '{substance}' -> canonical: '{canonical}'")

    # Test UI choices
    print("\n4. Substance choices for UI:")
    for substance, cls, display in get_substance_choices()[:5]:
        print(f"  {substance} -> {display}")
    print(f"  ... ({len(get_substance_choices())} total choices)")

    print("\n5. Class distribution:")
    for cls in sorted(get_all_classes()):
        substances = get_all_substances_for_class(cls)
        print(f"  Class {cls}: {len(substances)} substances")
