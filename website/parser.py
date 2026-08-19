"""
Question file parser. Supports CSV, TXT, PDF (text-based), and DOCX.

Expected format for TXT / DOCX / PDF:
    1. Question text here
    A) Option A
    B) Option B
    C) Option C
    D) Option D
    Answer: C

    2. Next question...

Expected format for CSV (header row required):
    question,option_a,option_b,option_c,option_d,answer
    What is 2+2?,1,2,3,4,4

All questions MUST have an answer key — upload will be rejected otherwise.
"""

import re
import csv
import io


def parse_questions_from_file(file_obj, extension):
    """
    Entry point. Returns a list of question dicts:
        [{'question': str, 'options': [str, str, str, str], 'answer': str}, ...]
    Raises ValueError with a human-readable message on any format error.
    """
    ext = extension.lower().lstrip('.')

    if ext == 'csv':
        content = file_obj.read().decode('utf-8-sig', errors='replace')
        return _parse_csv(content)
    elif ext == 'txt':
        content = file_obj.read().decode('utf-8-sig', errors='replace')
        return _parse_text(content)
    elif ext == 'pdf':
        return _parse_pdf(file_obj)
    elif ext == 'docx':
        return _parse_docx(file_obj)
    else:
        raise ValueError(f'Unsupported file type ".{ext}". Allowed: PDF, DOCX, TXT, CSV.')


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _parse_csv(content):
    reader = csv.DictReader(io.StringIO(content))

    if not reader.fieldnames:
        raise ValueError('CSV file is empty or has no header row.')

    normalized = {c.strip().lower(): c for c in reader.fieldnames}
    required = {'question', 'option_a', 'option_b', 'option_c', 'option_d', 'answer'}
    missing = required - set(normalized.keys())
    if missing:
        raise ValueError(
            f'CSV is missing required columns: {", ".join(sorted(missing))}.\n'
            f'Expected header: question, option_a, option_b, option_c, option_d, answer'
        )

    questions = []
    for i, row in enumerate(reader, start=2):
        q = row.get(normalized['question'], '').strip()
        opts = [
            row.get(normalized['option_a'], '').strip(),
            row.get(normalized['option_b'], '').strip(),
            row.get(normalized['option_c'], '').strip(),
            row.get(normalized['option_d'], '').strip(),
        ]
        ans = row.get(normalized['answer'], '').strip()

        if not q:
            raise ValueError(f'Row {i}: question text is empty.')
        if any(not o for o in opts):
            raise ValueError(f'Row {i}: one or more options (A–D) are empty.')
        if not ans:
            raise ValueError(
                f'Row {i}: answer is empty.\n'
                'Every question must have an answer key — upload rejected.'
            )
        if ans not in opts:
            raise ValueError(
                f'Row {i}: answer "{ans}" does not match any option.\n'
                'The answer column must contain the exact text of one of the four options.'
            )

        questions.append({'question': q, 'options': opts, 'answer': ans})

    if not questions:
        raise ValueError('No questions found in CSV. Make sure there are data rows below the header.')

    return questions


# ---------------------------------------------------------------------------
# Text-format parser (used by TXT, PDF, DOCX after extracting raw text)
# ---------------------------------------------------------------------------

def _parse_text(content):
    questions = []
    current = None

    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        q_match = re.match(r'^\d+[.)]\s+(.+)$', line)
        opt_match = re.match(r'^([A-D])[.)]\s+(.+)$', line, re.IGNORECASE)
        ans_match = re.match(r'^(?:Answer|Ans)\s*[:\-]\s*([A-D])$', line, re.IGNORECASE)

        if q_match:
            if current is not None:
                if current['answer'] is None:
                    raise ValueError(
                        f'Question "{current["question"][:50]}" has no Answer line.\n'
                        'Add "Answer: X" (where X is A, B, C, or D) after each question.'
                    )
                questions.append(current)
            current = {'question': q_match.group(1).strip(), 'options': [], 'answer': None}

        elif opt_match and current is not None:
            current['options'].append(opt_match.group(2).strip())

        elif ans_match and current is not None:
            if len(current['options']) != 4:
                raise ValueError(
                    f'Question "{current["question"][:50]}" has {len(current["options"])} option(s). '
                    'Expected exactly 4 (A, B, C, D).'
                )
            idx = ord(ans_match.group(1).upper()) - ord('A')
            current['answer'] = current['options'][idx]

    if current is not None:
        if current['answer'] is None:
            raise ValueError(
                f'Question "{current["question"][:50]}" has no Answer line.\n'
                'Add "Answer: X" after every question.'
            )
        questions.append(current)

    if not questions:
        raise ValueError(
            'No questions found. Check that the file follows the required format.\n'
            'Questions must start with a number (e.g. "1."), options with a letter (e.g. "A)"), '
            'and each question must end with "Answer: X".'
        )

    return questions


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------

def _parse_pdf(file_obj):
    try:
        import pdfplumber
    except ImportError:
        raise ValueError('PDF support requires pdfplumber. Run: pip install pdfplumber')

    try:
        parts = []
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
        content = '\n'.join(parts)
    except Exception as e:
        raise ValueError(f'Could not read PDF: {e}')

    if not content.strip():
        raise ValueError(
            'No text could be extracted from this PDF. '
            'It may be a scanned image — convert to a text-based PDF or use DOCX/TXT instead.'
        )

    return _parse_text(content)


# ---------------------------------------------------------------------------
# DOCX parser
# ---------------------------------------------------------------------------

def _parse_docx(file_obj):
    try:
        from docx import Document
    except ImportError:
        raise ValueError('DOCX support requires python-docx. Run: pip install python-docx')

    try:
        doc = Document(file_obj)
        content = '\n'.join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise ValueError(f'Could not read DOCX file: {e}')

    return _parse_text(content)
