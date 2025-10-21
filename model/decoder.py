import torch


class GreedyDecoder:
    def __init__(self, text_transform):
        self.text_transform = text_transform
        self.blank_idx = text_transform.blank_idx
    def decode(self, output):
        arg_maxes = torch.argmax(output, dim=2).transpose(0, 1)
        decodes = []
        for args in arg_maxes:
            decode = []
            for i, index in enumerate(args):
                if index != self.blank_idx:
                    if i != 0 and index == args[i-1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.text_transform.int_to_text(decode))
        return decodes


class BeamCTCDecoder:
    def __init__(self, text_transform, beam_width=10, blank_idx=None):
        self.text_transform = text_transform
        self.beam_width = beam_width
        self.blank_idx = blank_idx if blank_idx is not None else text_transform.blank_idx

    def decode(self, probs_seq):
        T, S = probs_seq.size()

        beams = [([], self.blank_idx, 0.0)]

        for t in range(T):
            next_beams = {}
            for prefix, last_char, prob in beams:
                for s in range(S):
                    p = probs_seq[t, s].item()
                    new_prefix = list(prefix)
                    if s == self.blank_idx:
                        key = (tuple(new_prefix), self.blank_idx)
                    else:
                        if last_char == s:
                            key = (tuple(new_prefix), s)
                        else:
                            new_prefix.append(s)
                            key = (tuple(new_prefix), s)
                    new_prob = prob + p
                    if key in next_beams:
                        old_prob = next_beams[key][2]
                        next_beams[key] = (list(key[0]), key[1], torch.logaddexp(torch.tensor(old_prob), torch.tensor(new_prob)).item())
                    else:
                        next_beams[key] = (list(key[0]), key[1], new_prob)
            beams = sorted(next_beams.values(), key=lambda x: x[2], reverse=True)[:self.beam_width]

        best_prefix, _, _ = beams[0]
        decoded = []
        prev = None
        for idx in best_prefix:
            if idx != self.blank_idx and idx != prev:
                decoded.append(idx)
            prev = idx
        return self.text_transform.int_to_text(decoded)

    def batch_decode(self, output):
        """
        output: (time, batch, num_classes) — log probs
        Returns: list of decoded strings
        """
        batch_size = output.size(1)
        results = []
        for b in range(batch_size):
            probs_seq = output[:, b, :]
            results.append(self.decode(probs_seq))
        return results
