from ..simdif import Metric, METRICS, to_list


def _to_soundex_char(element) -> str | None:
    try:
        s = str(element)
        if len(s) == 1 and s.isalpha():
            return s.upper()
    except Exception:
        pass
    return None
    

def _get_soundex(a, length=3):
    a = to_list(a)
    if len(a) == 0:
        return None, ""
    first = a[0]
    mapping = {'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3', 'L': '4', 'MN': '5', 'R': '6'}
    digits = ""
    last_digit = ""
    for x in a[1:]:
        c = _to_soundex_char(x)  # returns uppercase letter or None
        if c is None:
            continue
        for chars, digit in mapping.items():
            if c in chars:
                if digit != last_digit:
                    digits += digit
                    last_digit = digit
                break
        if len(digits) >= length:
            break
    return first, digits.ljust(length, '0')


def info_soundex() -> str:
    return """
Soundex Phonetic Algorithm
--------------------------
The standard American Soundex algorithm. It hashes words to a 4-character 
code based on their phonetic sound.

Algorithm:
    1. Retain the first letter.
    2. Drop A, E, I, O, U, Y, W, H, and anything that is not a letter
    3. Replace consonants with digits:
       B, F, P, V -> 1
       C, G, J, K, Q, S, X, Z -> 2
       D, T -> 3
       L -> 4
       M, N -> 5
       R -> 6
    4. Collapse adjacent identical digits.
    5. Truncate or pad with zeros to reach 4 characters.

Range: {0, 1}
    1 = Phonetically identical (same code)
    0 = Phonetically different
    """.strip()


def explain_soundex(a, b, **kwargs) -> str:
    length = kwargs.get('length', 3)
    first_a, code_a = _get_soundex(a, length)
    first_b, code_b = _get_soundex(b, length)
    match = 0.0
    if first_a is not None and first_b is not None and first_a == first_b and code_a == code_b:
        match = 1.0
    return f"""
A: {a} -> Soundex: {first_a}{code_a}
B: {b} -> Soundex: {first_b}{code_b}
Result: {match}
    """.strip()


@Metric
def sim_soundex(a, b, **kwargs) -> float:
    length = kwargs.get('length', 3)
    first_a, code_a = _get_soundex(a, length)
    first_b, code_b = _get_soundex(b, length)
    if first_a is None and first_b is None:
        return 1.0
    if first_a is None or first_b is None:
        return 0.0
    fa = _to_soundex_char(first_a)
    fb = _to_soundex_char(first_b)
    if fa is not None and fb is not None:
        if fa != fb:
            return 0.0
    else:
        if first_a != first_b:
            return 0.0
    return 1.0 if code_a == code_b else 0.0


@Metric
def dif_soundex(a, b, **kwargs) -> float:
    return 1.0 - sim_soundex(a, b, **kwargs)


METRICS['soundex'] = {
    'class': 'sequence',
    'default': 'sim',
    'sim': sim_soundex,
    'dif': dif_soundex,
    'info': info_soundex,
    'explain': explain_soundex,
}