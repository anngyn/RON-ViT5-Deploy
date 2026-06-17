"""OCR noise generator for Vietnamese receipts."""
import random
import re
import unicodedata


class OCRNoiseGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        self.char_map = {
            '0': 'O', 'O': '0',
            '1': 'l', 'l': '1',
            '5': 'S', 'S': '5',
            '8': 'B', 'B': '8'
        }

    def remove_accents(self, text):
        """Remove Vietnamese accents (diacritics)."""
        text = text.replace('đ', 'd').replace('Đ', 'D')
        normalized = unicodedata.normalize('NFD', text)
        return ''.join(ch for ch in normalized if unicodedata.category(ch) != 'Mn')

    def apply_char_confusion(self, text, prob=0.1):
        """Confuse similar-looking characters."""
        chars = list(text)
        for i in range(len(chars)):
            if chars[i] in self.char_map and random.random() < prob:
                chars[i] = self.char_map[chars[i]]
        return ''.join(chars)

    def corrupt_money(self, text, prob=0.3):
        """Corrupt Vietnamese currency patterns."""
        pattern = r'(\d+[.,]?\d*)đ?'

        def replace_fn(match):
            if random.random() > prob:
                return match.group(0)
            num = match.group(1).replace('.', ' ').replace(',', ' ')
            num = self.apply_char_confusion(num, prob=0.3)
            return num + ' d'

        return re.sub(pattern, replace_fn, text)

    def corrupt_date(self, text, prob=0.3):
        """Corrupt date patterns."""
        pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'

        def replace_fn(match):
            if random.random() > prob:
                return match.group(0)
            return self.apply_char_confusion(match.group(1), prob=0.4)

        return re.sub(pattern, replace_fn, text)

    def token_operations(self, text, del_prob=0.05, ins_prob=0.03):
        """Random token deletion/insertion."""
        tokens = text.split()
        new_tokens = []
        for token in tokens:
            if random.random() >= del_prob:
                new_tokens.append(token)
                if random.random() < ins_prob:
                    new_tokens.append(random.choice(['***', '---', 'XXX']))
        return ' '.join(new_tokens)

    def shuffle_lines(self, text, prob=0.2):
        """Shuffle lines/sentences (OCR ordering errors)."""
        lines = text.split('\n') if '\n' in text else text.split('. ')
        if len(lines) > 1 and random.random() < prob:
            random.shuffle(lines)
        return ' '.join(lines)

    def apply_noise(self, text, noise_type='mixed', level=2):
        """
        Apply noise to text.

        Args:
            text: input text
            noise_type: 'mixed', 'accent', 'char', 'money', 'date', 'token_ops', 'line_shuffle'
            level: 1=light, 2=medium, 3=heavy

        Returns:
            noisy text
        """
        scale = {1: 0.5, 2: 1.0, 3: 1.5}[level]

        if noise_type == 'accent':
            return self.remove_accents(text)
        elif noise_type == 'char':
            return self.apply_char_confusion(text, prob=0.1 * scale)
        elif noise_type == 'money':
            return self.corrupt_money(text, prob=0.3 * scale)
        elif noise_type == 'date':
            return self.corrupt_date(text, prob=0.3 * scale)
        elif noise_type == 'token_ops':
            return self.token_operations(text, del_prob=0.05 * scale, ins_prob=0.03 * scale)
        elif noise_type == 'line_shuffle':
            return self.shuffle_lines(text, prob=0.2 * scale)
        elif noise_type == 'mixed':
            # Apply multiple noise types
            if random.random() < 0.3:
                text = self.remove_accents(text)
            text = self.apply_char_confusion(text, prob=0.08 * scale)
            text = self.corrupt_money(text, prob=0.25 * scale)
            text = self.corrupt_date(text, prob=0.25 * scale)
            text = self.token_operations(text, del_prob=0.04 * scale, ins_prob=0.02 * scale)
            return text
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
