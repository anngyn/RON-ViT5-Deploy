"""OCR noise generator for Vietnamese receipts."""
import random
import re
import unicodedata


RECEIPT_VQA_NOISE_TYPES = [
    "accent_removal",
    "tone_confusion",
    "vowel_diacritic_confusion",
    "dd_confusion",
    "character_confusion",
    "glyph_confusion",
    "character_deletion",
    "token_deletion",
    "line_shuffle",
    "token_split",
    "money_noise",
    "date_noise",
    "code_noise",
    "mixed_noise",
]


NOISE_ALIASES = {
    "accent": "accent_removal",
    "char": "character_confusion",
    "money": "money_noise",
    "date": "date_noise",
    "token_ops": "token_deletion",
    "line_shuffle": "line_shuffle",
    "mixed": "mixed_noise",
}


class OCRNoiseGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.char_map = {
            "0": "O", "O": "0",
            "1": "l", "l": "1", "I": "1",
            "5": "S", "S": "5",
            "8": "B", "B": "8",
            "2": "Z", "Z": "2",
        }
        self.glyph_pairs = [
            ("rn", "m"),
            ("cl", "d"),
            ("vv", "w"),
        ]
        self.tone_groups = [
            "aàáảãạ", "ăằắẳẵặ", "âầấẩẫậ",
            "eèéẻẽẹ", "êềếểễệ",
            "iìíỉĩị",
            "oòóỏõọ", "ôồốổỗộ", "ơờớởỡợ",
            "uùúủũụ", "ưừứửữự",
            "yỳýỷỹỵ",
            "AÀÁẢÃẠ", "ĂẰẮẲẴẶ", "ÂẦẤẨẪẬ",
            "EÈÉẺẼẸ", "ÊỀẾỂỄỆ",
            "IÌÍỈĨỊ",
            "OÒÓỎÕỌ", "ÔỒỐỔỖỘ", "ƠỜỚỞỠỢ",
            "UÙÚỦŨỤ", "ƯỪỨỬỮỰ",
            "YỲÝỶỸỴ",
        ]
        self.vowel_groups = [
            "aăâ", "AĂÂ",
            "eê", "EÊ",
            "oôơ", "OÔƠ",
            "uư", "UƯ",
        ]

    def _scale(self, level):
        if level not in (1, 2, 3):
            raise ValueError(f"Unsupported noise level: {level}")
        return {1: 0.5, 2: 1.0, 3: 1.6}[level]

    def _normalize_noise_type(self, noise_type):
        return NOISE_ALIASES.get(noise_type, noise_type)

    def remove_accents(self, text):
        text = text.replace("đ", "d").replace("Đ", "D")
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def apply_tone_confusion(self, text, prob):
        chars = list(text)
        for idx, ch in enumerate(chars):
            for group in self.tone_groups:
                if ch in group and len(group) > 1 and self.rng.random() < prob:
                    repl = self.rng.choice([cand for cand in group if cand != ch])
                    chars[idx] = repl
                    break
        return "".join(chars)

    def apply_vowel_diacritic_confusion(self, text, prob):
        chars = list(text)
        for idx, ch in enumerate(chars):
            base = self.remove_accents(ch)
            for group in self.vowel_groups:
                if base in group and self.rng.random() < prob:
                    repl_base = self.rng.choice([cand for cand in group if cand != base] or [base])
                    chars[idx] = repl_base
                    break
        return "".join(chars)

    def apply_dd_confusion(self, text, prob):
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch in {"đ", "Đ"} and self.rng.random() < prob:
                chars[idx] = "d" if ch == "đ" else "D"
        return "".join(chars)

    def apply_char_confusion(self, text, prob=0.1):
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch in self.char_map and self.rng.random() < prob:
                chars[idx] = self.char_map[ch]
        return "".join(chars)

    def apply_glyph_confusion(self, text, prob):
        result = text
        for left, right in self.glyph_pairs:
            if left in result and self.rng.random() < prob:
                result = result.replace(left, right, 1)
            if right in result and self.rng.random() < prob:
                result = result.replace(right, left, 1)
        return result

    def apply_char_deletion(self, text, prob):
        chars = []
        for ch in text:
            if ch.isspace() or self.rng.random() >= prob:
                chars.append(ch)
        return "".join(chars) or text

    def apply_token_deletion(self, text, del_prob=0.05, ins_prob=0.0):
        tokens = text.split()
        new_tokens = []
        for token in tokens:
            if self.rng.random() >= del_prob:
                new_tokens.append(token)
                if ins_prob > 0 and self.rng.random() < ins_prob:
                    new_tokens.append(self.rng.choice(["***", "---", "XXX"]))
        return " ".join(new_tokens) if new_tokens else text

    def apply_line_shuffle(self, text, prob):
        lines = text.split("\n") if "\n" in text else [seg for seg in text.split(". ") if seg]
        if len(lines) > 1 and self.rng.random() < prob:
            self.rng.shuffle(lines)
            return " ".join(lines)
        return text

    def apply_token_split(self, text, prob):
        tokens = []
        for token in text.split():
            if len(token) >= 4 and self.rng.random() < prob:
                split_at = max(1, min(len(token) - 1, len(token) // 2))
                tokens.extend([token[:split_at], token[split_at:]])
            elif len(token) >= 5 and token.isdigit() and self.rng.random() < prob:
                split_at = len(token) // 2
                tokens.extend([token[:split_at], token[split_at:]])
            else:
                tokens.append(token)
        return " ".join(tokens)

    def corrupt_money(self, text, prob):
        pattern = re.compile(r"(\d[\d., ]*)(đ|d)?", flags=re.IGNORECASE)

        def replace_fn(match):
            if self.rng.random() > prob:
                return match.group(0)
            num = match.group(1).replace(".", " ").replace(",", " ")
            num = self.apply_char_confusion(num, prob=0.35)
            return f"{num} d"

        return pattern.sub(replace_fn, text)

    def corrupt_date(self, text, prob):
        pattern = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")

        def replace_fn(match):
            if self.rng.random() > prob:
                return match.group(0)
            return self.apply_char_confusion(match.group(1), prob=0.45)

        return pattern.sub(replace_fn, text)

    def corrupt_codes(self, text, prob):
        pattern = re.compile(r"\b[A-Z0-9]{5,}\b")

        def replace_fn(match):
            if self.rng.random() > prob:
                return match.group(0)
            token = self.apply_char_confusion(match.group(0), prob=0.35)
            token = self.apply_char_deletion(token, prob=0.08)
            return token

        return pattern.sub(replace_fn, text)

    def apply_noise(self, text, noise_type="mixed_noise", level=2):
        scale = self._scale(level)
        noise_type = self._normalize_noise_type(noise_type)

        if noise_type == "accent_removal":
            return self.remove_accents(text)
        if noise_type == "tone_confusion":
            return self.apply_tone_confusion(text, prob=0.08 * scale)
        if noise_type == "vowel_diacritic_confusion":
            return self.apply_vowel_diacritic_confusion(text, prob=0.07 * scale)
        if noise_type == "dd_confusion":
            return self.apply_dd_confusion(text, prob=0.18 * scale)
        if noise_type == "character_confusion":
            return self.apply_char_confusion(text, prob=0.1 * scale)
        if noise_type == "glyph_confusion":
            return self.apply_glyph_confusion(text, prob=0.12 * scale)
        if noise_type == "character_deletion":
            return self.apply_char_deletion(text, prob=0.03 * scale)
        if noise_type == "token_deletion":
            return self.apply_token_deletion(text, del_prob=0.05 * scale, ins_prob=0.0)
        if noise_type == "line_shuffle":
            return self.apply_line_shuffle(text, prob=0.2 * scale)
        if noise_type == "token_split":
            return self.apply_token_split(text, prob=0.12 * scale)
        if noise_type == "money_noise":
            return self.corrupt_money(text, prob=0.3 * scale)
        if noise_type == "date_noise":
            return self.corrupt_date(text, prob=0.3 * scale)
        if noise_type == "code_noise":
            return self.corrupt_codes(text, prob=0.25 * scale)
        if noise_type == "mixed_noise":
            if self.rng.random() < 0.35:
                text = self.remove_accents(text)
            if self.rng.random() < 0.55:
                text = self.apply_tone_confusion(text, prob=0.05 * scale)
            text = self.apply_char_confusion(text, prob=0.08 * scale)
            text = self.apply_glyph_confusion(text, prob=0.08 * scale)
            text = self.corrupt_money(text, prob=0.2 * scale)
            text = self.corrupt_date(text, prob=0.2 * scale)
            text = self.corrupt_codes(text, prob=0.15 * scale)
            text = self.apply_token_deletion(text, del_prob=0.04 * scale, ins_prob=0.02 * scale)
            return self.apply_token_split(text, prob=0.06 * scale)

        raise ValueError(f"Unknown noise type: {noise_type}")
