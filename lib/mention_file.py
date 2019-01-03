'''
I/O wrappers for mention files (used for streaming feature generation)
'''

import codecs

class EmbeddedMention:
    def __init__(self, CUI, mention_repr, context_repr, candidates, ID=None):
        self.CUI = CUI
        self.mention_repr = mention_repr
        self.context_repr = context_repr
        self.candidates = candidates
        self.ID = ID

def write(mentions, outf, encoding='utf-8'):
    max_ID = 0
    for m in mentions:
        if not m.ID is None and m.ID > max_ID:
            max_ID = m.ID

    fmt_str = 'BIN'

    with codecs.open(outf, 'w', encoding) as stream:
        stream.write('%s\n' % fmt_str)
        for m in mentions:
            if m.ID is None:
                m.ID = max_ID
                max_ID += 1
            stream.write('%s\n' % '\t'.join([
                str(m.ID),
                'None' if not m.mention_repr else (' '.join([str(f) for f in m.mention_repr])),
                ' '.join([str(f) for f in m.context_repr]),
                m.CUI,
                '||'.join(m.candidates)
            ]))

def read(mentionf, encoding='utf-8'):
    mentions = []
    with codecs.open(mentionf, 'r', encoding) as stream:
        fmt = stream.read(3)
        stream.seek(0)
        if not fmt in ['BIN']:
            fmt = 'BIN'
        stream.readline()

        for line in stream:
            chunks = [s.strip() for s in line.split('\t')]
            _id = int(chunks[0])
            if chunks[1] == 'None':
                mention_repr = None
            else:
                mention_repr = [float(f) for f in chunks[1].split()]
            context_repr = [float(f) for f in chunks[2].split()]
            CUI = chunks[3]
            candidates = chunks[4].split('||')
            mentions.append(EmbeddedMention(
                CUI, mention_repr, context_repr, candidates, ID=_id
            ))
    return mentions
