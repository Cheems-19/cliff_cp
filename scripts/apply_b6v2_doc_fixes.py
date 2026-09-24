# -*- coding: utf-8 -*-
"""Apply B6-v2 data corrections + insert Figure 15 into the four manuscripts.

Every replacement asserts exactly-one occurrence of the old string; every
file is re-read after writing to verify old strings are gone and new strings
are present. (Direct file I/O: the Edit tool has silently no-op'd on these
files before.)
"""
import io
import sys

ROOT = r"C:/Users/Administrator/Desktop/小论文/cliff_cp"


def load(path):
    with io.open(f"{ROOT}/{path}", "r", encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")   # normalise CRLF -> LF


def save(path, s):
    with io.open(f"{ROOT}/{path}", "w", encoding="utf-8", newline="") as fh:
        fh.write(s.replace("\n", "\r\n"))        # restore CRLF


def rep(s, old, new, tag, path):
    n = s.count(old)
    assert n == 1, f"[{path}] {tag}: expected 1 occurrence, found {n}\n>>> {old[:90]}"
    s = s.replace(old, new)
    print(f"  ok {tag}")
    return s


def insert_after(s, anchor, block, tag, path):
    n = s.count(anchor)
    assert n == 1, f"[{path}] {tag}: anchor count {n}\n>>> {anchor[:90]}"
    s = s.replace(anchor, anchor + block)
    print(f"  ok {tag}")
    return s


# ---------------------------------------------------------------- v1 (EN)
p = "manuscript_en_v1.md"
s = load(p)
s = rep(s,
        "small (about 3 to 41 molecules per seed), so",
        "small (on average 1.5 to 37 molecules per seed, and some ClinTox\n"
        "  seeds contain none), so",
        "v1 3-to-41 -> measured means", p)
s = rep(s,
        "(label-cliff coverage at most +5.3 pp on Tox21-NR-AR)",
        "(label-cliff coverage at most +5.4 pp on Tox21-NR-AR)",
        "v1 nnsim +5.3 -> +5.4", p)
s = rep(s,
        "- **Coverage failure again concentrates on the cliff group**: across",
        "- **Coverage failure again concentrates on the cliff group** (Figure 15): across",
        "v1 bullet cites Figure 15", p)
fig15_v1 = """
![Figure 15](fig15_classify.png)

**Figure 15.** The classification transfer at a glance (10 seeds per benchmark; \u03b1 = 0.10; Tox21 =
Tox21-NR-AR; cliff groups average 1.5 to 37 molecules per seed). **(a)** Per-seed label-cliff versus
non-cliff coverage of global split CP: grey lines pair the two neighbourhoods of the same seed,
filled discs mark the cliff mean and open discs the non-cliff mean; the tags below the axis give the
mean neighbourhood gap (pp) and the number of seeds with a positive gap. **(b)** Change in
label-cliff coverage versus global for the two Mondrian arms: the predicted class is an exact
no-op; nearest-neighbour similarity recovers at most +5.4 pp. **(c)** Label-cliff coverage versus
mean label-cliff set size: APS (open squares) reaches nominal coverage only by returning both
labels in every set (mean size 1.90-2.00), while global CP (filled discs) mostly returns empty or
singleton sets on cliffs (empty sets counted as size 0).
"""
s = insert_after(s, "same mechanism as cliffB in regression.\n", fig15_v1,
                 "v1 insert Figure 15 block", p)
save(p, s)

# ---------------------------------------------------------------- v2 (EN polished)
p = "manuscript_en_v2_polished.md"
s = load(p)
s = rep(s,
        "The label-cliff groups are small (about 3 to 41 molecules per seed), so",
        "The label-cliff groups are small (on average 1.5 to 37 molecules per seed, "
        "and some ClinTox seeds contain none), so",
        "v2 3-to-41 -> measured means", p)
s = rep(s,
        "- **Coverage failure again concentrates on the cliff group**: across all five",
        "- **Coverage failure again concentrates on the cliff group** (Figure 15): across all five",
        "v2 bullet cites Figure 15", p)
fig15_v2 = """
![Figure 15](fig15_classify.png)

**Figure 15.** The classification transfer at a glance (10 seeds per benchmark; alpha = 0.10; Tox21 = Tox21-NR-AR; cliff groups average 1.5 to 37 molecules per seed). **(a)** Per-seed label-cliff versus non-cliff coverage of global split CP: grey lines pair the two neighbourhoods of the same seed, filled discs mark the cliff mean and open discs the non-cliff mean; the tags below the axis give the mean neighbourhood gap (pp) and the number of seeds with a positive gap. **(b)** Change in label-cliff coverage versus global for the two Mondrian arms: the predicted class is an exact no-op; nearest-neighbour similarity recovers at most +5.4 pp. **(c)** Label-cliff coverage versus mean label-cliff set size: APS (open squares) reaches nominal coverage only by returning both labels in every set (mean size 1.90 to 2.00), while global CP (filled discs) mostly returns empty or singleton sets on cliffs (empty sets counted as size 0).
"""
s = insert_after(s, "the same mechanism as cliffB in regression.\n\n", fig15_v2,
                 "v2 insert Figure 15 block", p)
save(p, s)

# ---------------------------------------------------------------- SI
p = "SI_v1.md"
s = load(p)
s = rep(s,
        "random 60/20/20 splits, 5 seeds) on three MoleculeNet benchmarks",
        "random 60/20/20 splits, 10 seeds) on five MoleculeNet benchmarks",
        "SI S10 5-seeds/3-benchmarks -> 10/5", p)
s = rep(s,
        "| Dataset | n | cliffs / seed | cov (global) |",
        "| Dataset | n | cliffs / seed (max) | cov (global) |",
        "SI S10 header marks max", p)
s = rep(s,
        "(label-cliff groups hold only 3\u201327 molecules per seed, so",
        "(label-cliff groups average only 1.5\u201337 molecules per seed, and some ClinTox "
        "seeds contain no cliff molecule, so",
        "SI 3-27 -> measured means", p)
s = rep(s,
        "improves label-cliff coverage by at most +5.3 pp",
        "improves label-cliff coverage by at most +5.4 pp",
        "SI nnsim +5.3 -> +5.4", p)
save(p, s)

# ---------------------------------------------------------------- CN
p = "\u8bba\u6587\u5168\u6587_v1.md"
s = load(p)
s = rep(s,
        "\uff08cliff \u8986\u76d6\u6700\u591a +5.3 pp\uff0cTox21-NR-AR\uff09",
        "\uff08cliff \u8986\u76d6\u6700\u591a +5.4 pp\uff0cTox21-NR-AR\uff09",
        "CN nnsim +5.3 -> +5.4", p)
s = rep(s,
        "\uff08BBBP/ClinTox \u4ec5 3\u20134 \u4e2a/\u79cd\u5b50\uff09",
        "\uff08\u6bcf\u79cd\u5b50\u5e73\u5747 BBBP 2.4 \u4e2a\u3001ClinTox 1.5 \u4e2a\uff0c"
        "ClinTox \u90e8\u5206\u79cd\u5b50\u65e0\u60ac\u5d16\u5206\u5b50\uff09",
        "CN 3-4-per-seed note -> measured means", p)
s = rep(s,
        "- **\u8986\u76d6\u5931\u6548\u540c\u6837\u96c6\u4e2d\u4e8e\u6807\u7b7e\u60ac\u5d16**\uff1a\u4e94\u4e2a\u6570\u636e\u96c6",
        "- **\u8986\u76d6\u5931\u6548\u540c\u6837\u96c6\u4e2d\u4e8e\u6807\u7b7e\u60ac\u5d16**\uff08\u56fe15\uff09\uff1a\u4e94\u4e2a\u6570\u636e\u96c6",
        "CN bullet cites Fig15", p)
fig15_cn = (
    "\n![\u56fe15](fig15_classify.png)\n"
    "\n"
    "**\u56fe15**\u3000\u5206\u7c7b\u6807\u7b7e\u60ac\u5d16\uff1a\u8986\u76d6\u574d\u7f29\u4e0e APS \u7684\u9000\u5316\u4fee\u590d"
    "\uff08\u6bcf\u4e2a\u57fa\u51c6 10 \u4e2a\u79cd\u5b50\uff0c\u03b1=0.10\uff1bTox21 \u5373\n"
    "Tox21-NR-AR\uff1b\u60ac\u5d16\u7ec4\u5e73\u5747\u6bcf\u79cd\u5b50 1.5~37 \u4e2a\u5206\u5b50\uff09\u3002"
    "**(a)** global \u5206\u88c2\u5171\u5f62\u4e0b\u9010\u79cd\u5b50\u7684\u6807\u7b7e\u60ac\u5d16\u4e0e\n"
    "\u975e\u60ac\u5d16\u8986\u76d6\uff1a\u7070\u7ebf\u8fde\u63a5\u540c\u4e00\u79cd\u5b50\u7684\u4e24\u4e2a\u90bb\u57df\uff0c"
    "\u5b9e\u5fc3\u5706\u4e3a\u60ac\u5d16\u5747\u503c\u3001\u7a7a\u5fc3\u5706\u4e3a\u975e\u60ac\u5d16\u5747\u503c\uff0c\u8f74\u4e0b\u6807\u7b7e\u4e3a\n"
    "\u5e73\u5747\u90bb\u57df\u5dee\u8ddd\uff08pp\uff09\u4e0e\u5dee\u8ddd\u4e3a\u6b63\u7684\u79cd\u5b50\u6570\uff1b"
    "**(b)** \u4e24\u4e2a Mondrian \u81c2\u76f8\u5bf9 global \u7684\u6807\u7b7e\u60ac\u5d16\u8986\u76d6\u53d8\u5316\uff1a\n"
    "\u9884\u6d4b\u7c7b\u662f\u7cbe\u786e\u7684\u65e0\u64cd\u4f5c\uff0c\u8fd1\u90bb\u76f8\u4f3c\u5ea6\u6700\u591a\u633d\u56de +5.4 pp\uff1b"
    "**(c)** \u6807\u7b7e\u60ac\u5d16\u8986\u76d6\u5bf9\u5e73\u5747\u6807\u7b7e\u60ac\u5d16\u96c6\u5408\u5927\u5c0f\uff1a\n"
    "APS\uff08\u7a7a\u5fc3\u65b9\u5757\uff09\u9760\u300c\u6bcf\u4e2a\u96c6\u5408\u90fd\u8fd4\u56de\u53cc\u6807\u7b7e\u300d"
    "\uff08\u5747\u503c 1.90~2.00\uff09\u8fbe\u5230\u540d\u4e49\u8986\u76d6\uff0cglobal CP\uff08\u5b9e\u5fc3\u5706\uff09\n"
    "\u5728\u60ac\u5d16\u4e0a\u5927\u591a\u8fd4\u56de\u7a7a\u96c6\u6216\u5355\u6807\u7b7e\u96c6\uff08\u7a7a\u96c6\u8ba1\u4e3a 0\uff09\u3002\n"
)
s = insert_after(s,
                 "\u4e0b\u8f7d\u7f13\u5b58\u4e8e\u670d\u52a1\u5668 `data/classification/`\u3002\uff09\n",
                 fig15_cn, "CN insert Fig15 block", p)
s = rep(s,
        "| \u56fe 14 | \u60ac\u5d16\u5bf9\u5b9e\u4f8b\uff08\u540c\u4e00\u6307\u7eb9\u3001\u540c\u4e00\u533a\u95f4\u3001\u4e00\u5931\u4e00\u8fbe\uff09 "
        "| `figures/fig14_cliff_pair.png` | 600 dpi \u2713 |",
        "| \u56fe 14 | \u60ac\u5d16\u5bf9\u5b9e\u4f8b\uff08\u540c\u4e00\u6307\u7eb9\u3001\u540c\u4e00\u533a\u95f4\u3001\u4e00\u5931\u4e00\u8fbe\uff09 "
        "| `figures/fig14_cliff_pair.png` | 600 dpi \u2713 |\n"
        "| \u56fe 15 | \u5206\u7c7b\u6807\u7b7e\u60ac\u5d16\uff1a\u8986\u76d6\u574d\u7f29\u4e0e APS \u9000\u5316\u4fee\u590d "
        "| `figures/fig15_classify.png` | 600 dpi \u2713 |",
        "CN figure table row", p)
save(p, s)

# --------------------------------------------------------- read-back verify
print("\n=== read-back verification ===")
checks = {
    "manuscript_en_v1.md": [
        ("about 3 to 41", 0), ("+5.3 pp on Tox21-NR-AR", 0),
        ("on average 1.5 to 37 molecules per seed", 1),
        ("+5.4 pp on Tox21-NR-AR", 1),
        ("![Figure 15](fig15_classify.png)", 1),
        ("**(c)** Label-cliff coverage versus", 1),
        ("cliff group** (Figure 15): across", 1),
    ],
    "manuscript_en_v2_polished.md": [
        ("about 3 to 41", 0), ("on average 1.5 to 37 molecules per seed", 1),
        ("![Figure 15](fig15_classify.png)", 1),
        ("cliff group** (Figure 15): across all five", 1),
    ],
    "SI_v1.md": [
        ("3\u201327", 0), ("+5.3 pp", 0), ("1.5\u201337 molecules per seed", 1),
        ("10 seeds) on five MoleculeNet benchmarks", 1),
        ("cliffs / seed (max)", 1), ("+5.4 pp", 1),
    ],
    "\u8bba\u6587\u5168\u6587_v1.md": [
        ("+5.3 pp\uff0cTox21-NR-AR", 0),
        ("\u4ec5 3\u20134 \u4e2a/\u79cd\u5b50", 0),
        ("\u6bcf\u79cd\u5b50\u5e73\u5747 BBBP 2.4 \u4e2a", 1),
        ("+5.4 pp\uff0cTox21-NR-AR", 1),
        ("![\u56fe15](fig15_classify.png)", 1),
        ("| \u56fe 15 |", 1),
        ("\uff08\u56fe15\uff09\uff1a\u4e94\u4e2a\u6570\u636e\u96c6", 1),
    ],
}
ok = True
for path, pats in checks.items():
    s = load(path)
    for pat, want in pats:
        got = s.count(pat)
        flag = "ok " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] {path}: {pat[:44]!r} -> {got} (want {want})")
print("\nALL VERIFIED" if ok else "\nVERIFICATION FAILED")
sys.exit(0 if ok else 1)
