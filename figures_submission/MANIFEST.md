# 投稿图件清单（figures_submission/）

规格：TIFF（**zlib / Adobe Deflate 无损压缩**，经 tifffile 写出）目标双栏宽 **17.4 cm**；**仅下采样，不上采样**。
另附**矢量 PDF**（由绘图脚本直接输出，线稿与文字可无损缩放，多数期刊首选）。

| 图 | TIFF | 大小 | 矢量 PDF | 像素 | 宽 (cm) | 有效 dpi | 建议栏位 |
|---|---|---|---|---|---|---|---|
| Figure 1 | `Fig1_coverage_gap.tif` | 389 KB | `Fig1_coverage_gap.pdf` | 4110x1605 | 17.4 | 600 | double column |
| Figure 2 | `Fig2_mechanism.tif` | 726 KB | `Fig2_mechanism.pdf` | 4110x2042 | 17.4 | 600 | double column |
| Figure 3 | `Fig3_ablation.tif` | 312 KB | `Fig3_ablation.pdf` | 3309x2716 | 14.01 | 600 | double column (natural size) |
| Figure 4 | `Fig4_frontier.tif` | 560 KB | `Fig4_frontier.pdf` | 4110x1691 | 17.4 | 600 | double column |
| Figure 5 | `Fig5_ad_blindness.tif` | 672 KB | `Fig5_ad_blindness.pdf` | 4110x1736 | 17.4 | 600 | double column |
| Figure 6 | `Fig6_ad_sensitivity.tif` | 852 KB | `Fig6_ad_sensitivity.pdf` | 4110x2069 | 17.4 | 600 | double column |
| Figure 7 | `Fig7_representation.tif` | 450 KB | `Fig7_representation.pdf` | 4110x2031 | 17.4 | 600 | double column |
| Figure 8 | `Fig8_threshold_sensitivity.tif` | 1033 KB | `Fig8_threshold_sensitivity.pdf` | 4110x2049 | 17.4 | 600 | double column |
| Figure 9 | `Fig9_method_ladder.tif` | 619 KB | `Fig9_method_ladder.pdf` | 4110x2474 | 17.4 | 600 | double column |
| Figure 10 | `Fig10_misalignment_vs_harm.tif` | 360 KB | `Fig10_misalignment_vs_harm.pdf` | 3788x2830 | 16.04 | 600 | double column (natural size) |
| Figure 11 | `Fig11_decision_impact.tif` | 400 KB | `Fig11_decision_impact.pdf` | 4110x1604 | 17.4 | 600 | double column |
| Figure 12 | `Fig12_transfer_diagnostic.tif` | 494 KB | `Fig12_transfer_diagnostic.pdf` | 4110x2604 | 17.4 | 600 | double column |
| Figure 13 | `Fig13_shift_spectrum.tif` | 710 KB | `Fig13_shift_spectrum.pdf` | 4110x1421 | 17.4 | 600 | double column |
| Figure 14 | `Fig14_cliff_pair.tif` | 490 KB | `Fig14_cliff_pair.pdf` | 4110x1701 | 17.4 | 600 | double column |
| Figure 15 | `Fig15_classify.tif` | 550 KB | `Fig15_classify.pdf` | 4110x1492 | 17.4 | 600 | double column |

> 说明：所有图均为 600 dpi 重绘；TIFF 已按双栏宽归一化（有效 dpi 见上表）。
> 若目标刊要求单栏（8.4 cm）或指定宽度，改 `DOUBLE_CM` 后以 `--export-only` 重跑本脚本即可（跳过重绘）。
> 注：本机 Pillow 12.3.0 的压缩 TIFF 写出会崩溃，故 TIFF 由 tifffile 生成；LZW 需 `imagecodecs`，未安装时用 zlib（期刊同样接受）。
