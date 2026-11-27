# SD-08 local real-data smoke record

Recorded 2026-11-27 from the opt-in release smoke run using locally acquired Dryad recordings.
Every session below was checked in both trial and fixed-time views with 20 ms bins, three outer
grouped folds (seed 42), and two grouped inner folds. All native bin widths were 1 ms.
The smoke test replayed each manifest deterministically and found zero raw-sample span overlap
between outer fit/held-out and inner fit/validation roles.

Fold notation: outer lists `fit/held-out` group counts for folds 0, 1, 2. Inner lists, for each
outer fold in order, the two inner `fit/validation` group-count pairs.

| Dataset | Session ID | Views | Target | Eligible groups (IDs) | Outer fit/held-out | Inner fit/validation |
|---|---|---|---|---:|---|---|
| chewie | `Chewie_20160927_001` | trial, fixed | `curs_v` | 241 (`0-240`) | `160/81,161/80,161/80` | `80/80+80/80;80/81+81/80;80/81+81/80` |
| chewie | `Chewie_20160928_001` | trial, fixed | `curs_v` | 246 (`0-245`) | `164/82,164/82,164/82` | `82/82+82/82;82/82+82/82;82/82+82/82` |
| chewie | `Chewie_20160929_001` | trial, fixed | `curs_v` | 253 (`0-252`) | `168/85,169/84,169/84` | `84/84+84/84;84/85+85/84;84/85+85/84` |
| chewie | `Chewie_20160930_001` | trial, fixed | `curs_v` | 244 (`0-243`) | `162/82,163/81,163/81` | `81/81+81/81;81/82+82/81;81/82+82/81` |
| chewie | `Chewie_20161006_001` | trial, fixed | `curs_v` | 251 (`0-250`) | `167/84,167/84,168/83` | `83/84+84/83;83/84+84/83;84/84+84/84` |
| chewie | `Chewie_20161007_001` | trial, fixed | `curs_v` | 213 (`0-212`) | `142/71,142/71,142/71` | `71/71+71/71;71/71+71/71;71/71+71/71` |
| chewie | `Chewie_20161011_001` | trial, fixed | `curs_v` | 239 (`0-238`) | `159/80,159/80,160/79` | `79/80+80/79;79/80+80/79;80/80+80/80` |
| chewie | `Chewie_20161012_001` | trial, fixed | `curs_v` | 283 (`0-282`) | `188/95,189/94,189/94` | `94/94+94/94;94/95+95/94;94/95+95/94` |
| chewie | `Chewie_20161013_001` | trial, fixed | `curs_v` | 271 (`0-270`) | `180/91,181/90,181/90` | `90/90+90/90;90/91+91/90;90/91+91/90` |
| chewie | `Chewie_20161102_001` | trial, fixed | `curs_v` | 260 (`0-259`) | `173/87,173/87,174/86` | `86/87+87/86;86/87+87/86;87/87+87/87` |
| chewie | `Chewie_20161103_001` | trial, fixed | `curs_v` | 243 (`0-242`) | `162/81,162/81,162/81` | `81/81+81/81;81/81+81/81;81/81+81/81` |
| chewie | `Chewie_20161104_001` | trial, fixed | `curs_v` | 242 (`0-241`) | `161/81,161/81,162/80` | `80/81+81/80;80/81+81/80;81/81+81/81` |
| jango | `Jango_20150730_001` | trial, fixed | `force` | 186 (`0-185`) | `124/62,124/62,124/62` | `62/62+62/62;62/62+62/62;62/62+62/62` |
| jango | `Jango_20150731_001` | trial, fixed | `force` | 204 (`0-203`) | `136/68,136/68,136/68` | `68/68+68/68;68/68+68/68;68/68+68/68` |
| jango | `Jango_20150809_001` | trial, fixed | `force` | 226 (`0-225`) | `150/76,151/75,151/75` | `75/75+75/75;75/76+76/75;75/76+76/75` |
| jango | `Jango_20150827_001` | trial, fixed | `force` | 227 (`0-226`) | `151/76,151/76,152/75` | `75/76+76/75;75/76+76/75;76/76+76/76` |
| jango | `Jango_20150908_001` | trial, fixed | `force` | 214 (`0-213`) | `142/72,143/71,143/71` | `71/71+71/71;71/72+72/71;71/72+72/71` |
| jango | `Jango_20151029_001` | trial, fixed | `force` | 217 (`0-216`) | `144/73,145/72,145/72` | `72/72+72/72;72/73+73/72;72/73+73/72` |
| jango | `Jango_20151102_001` | trial, fixed | `force` | 218 (`0-217`) | `145/73,145/73,146/72` | `72/73+73/72;72/73+73/72;73/73+73/73` |

Fixed-time Jango Day-0 → Day-K exercise: `dataset=jango view=fixed day0=Jango_20150730_001 dayk=Jango_20151102_001 target=force day0_groups=186 dayk_groups=218 fold=0 target_bins=14927`. The zero-shot scorer opened 14,927 held-out target bins.

This is a protocol smoke check with a simple finite-target mean scorer, not a scientific model
comparison or a claim about predictive performance.
