def is_invalid_before_breakfast(value):
    return value == "[]" or not value or len(value.strip()) == 0