# -*- coding: utf-8 -*-
import io
P = r"C:/Users/Administrator/Desktop/小论文/cliff_cp/论文全文_v1.md"
s = io.open(P, encoding="utf-8").read().replace("\r\n", "\n")
for l in s.split("\n"):
    if "0.03" in l:
        print("=== full line (len %d) ===" % len(l))
        print(repr(l))
        print("=== around 0.03 ===")
        j = l.find("0.03")
        print(repr(l[max(0,j-60): j+90]))
