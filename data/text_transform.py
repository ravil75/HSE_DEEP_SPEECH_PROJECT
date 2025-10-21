class TextTransform:
    def __init__(self):
        self.char_map = {' ': 0, "'": 1}
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
            self.char_map[c] = i + 2
        self.index_map = {v: k for k, v in self.char_map.items()}
        self.blank_idx = len(self.char_map)
    def text_to_int(self, text):
        text = text.lower().strip()
        int_sequence = [self.char_map[c] for c in text if c in self.char_map]
        return int_sequence
    def int_to_text(self, labels):
        return ''.join([self.index_map.get(i, '') for i in labels if i != self.blank_idx])
