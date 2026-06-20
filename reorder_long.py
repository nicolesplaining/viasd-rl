#!/usr/bin/env python3
"""Reorder slides.tex (long deck): move the two Ablation frames and the Future-work
frame to AFTER the closing 'Thank you' slide (as backup/appendix). Only touches the tail."""
import re

s = open("slides.tex").read()
sep_re = re.compile(r'% ={5,}\n')

marker = r'\begin{frame}{The accuracy and speed frontier}'   # pareto
mi = s.index(marker)
cut = [m.start() for m in sep_re.finditer(s) if m.start() < mi][-1]
head, tail = s[:cut], s[cut:]

parts = sep_re.split(tail)        # parts[0]=='' ; then blocks in order
# expected: 1=pareto, 2=abl1, 3=abl2, 4=impact, 5=future(+closing+enddoc)
labels = [re.search(r'\\begin\{frame\}(?:\[[^\]]*\])?\{?([^}\n]*)', p) for p in parts]
order = [ (l.group(1)[:30] if l else repr(p[:20])) for l,p in zip(labels,parts) ]
print("BEFORE blocks:", order)

pareto, abl1, abl2, impact = parts[1], parts[2], parts[3], parts[4]
future_and_close = parts[5]
ci = future_and_close.index(r'\begin{frame}[plain]')
future = future_and_close[:ci]
closing_and_end = future_and_close[ci:]
ei = closing_and_end.index(r'\end{document}')
closing, enddoc = closing_and_end[:ei], closing_and_end[ei:]

SEP = "% =====================================================================\n"
new_tail = (SEP + pareto + SEP + impact + SEP + closing
            + SEP + abl1 + SEP + abl2 + SEP + future + enddoc)
open("slides.tex", "w").write(head + new_tail)

after = re.findall(r'\\begin\{frame\}(?:\[[^\]]*\])?\{?([^}\n]*)', head + new_tail)
print("AFTER frame order:")
for t in after: print("  -", t[:45])
