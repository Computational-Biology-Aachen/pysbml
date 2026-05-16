# SBML L3v2 Test-Case Divergence Tracker

**Spec:** SBML Level 3 Version 2 Core Release 2 (29 March 2019)
**Scope:** pysbml vs SBML L3v2 spec, measured against the 1692-case BioMD/SBML test suite (`tests/assets/`).
See [`implementations.md`](implementations.md) for decision-tree details and cross-library comparison.

---

## Divergence Flag Definitions

| Flag             | Meaning                                                               |
| ---------------- | --------------------------------------------------------------------- |
| `SPEC_SILENT`    | Test exercises behavior the spec leaves undefined or does not address |
| `SPEC_CONFLICT`  | Test or implementation contradicts a normative spec statement         |
| `SPEC_EXTENSION` | Implementation adds behavior clearly beyond what spec defines         |

---

## Divergence Index

| ID  | Flag             | Element         | Description                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --- | ---------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | `SPEC_SILENT`    | Species         | Neither `initialAmount` nor `initialConcentration` set: spec §4.6.4 says value is "unknown or from external source". pysbml injects 0.0 using a co-reactant heuristic to decide amount vs concentration. Tests: t676, t688 → amount; t1513 → concentration.                                                                                                                                                                               |
| D2  | `SPEC_SILENT`    | Species         | `_handle_amount_boundary_has_substance_units` (test 1123): source FIXME notes unexplained behavioral difference from non-boundary path. Spec does not differentiate these two paths. The handler derives `{k}_conc` and forces `rxn_had_compartment=True` for all containing reactions.                                                                                                                                                   |
| D3  | `SPEC_EXTENSION` | Reaction        | `fast=True` triggers full QSS reduction with conservation-law-based stoichiometry correction and initial condition projection. L3v2 spec §4.11 explicitly removed `fast`. pysbml retains for L3v1 compatibility.                                                                                                                                                                                                                          |
| D4  | `SPEC_SILENT`    | Reaction        | `rxn_had_compartment` heuristic: spec says kinetic law unit is extent/time. pysbml auto-detects traditional concentration/time laws (those containing a compartment symbol) and auto-corrects by removing the compartment from the kinetic law and multiplying stoichiometry by it. Not described in spec.                                                                                                                                |
| D5  | `SPEC_SILENT`    | RateRule        | Chain rule correction for dynamic compartment + rate rule on `hasOnlySubstanceUnits=false` species: `d(amount)/dt = d(conc)/dt · C + conc · dC/dt`. Spec §4.9.4 defines rate rule math as `d(variable)/dt` but does not specify reconciliation with a dynamic compartment.                                                                                                                                                                |
| D6  | `SPEC_SILENT`    | AlgebraicRule   | Floating variable identification and algebraic rule solving via `sympy.solve`. Spec requires exactly one undetermined variable but does not define the algorithm. May fail for nonlinear rules.                                                                                                                                                                                                                                           |
| D7  | `SPEC_EXTENSION` | Constraint      | Constraints are parsed but silently ignored at transform time; no runtime checking. Spec says interpreters *may* warn or halt on violation.                                                                                                                                                                                                                                                                                               |
| D8  | `SPEC_SILENT`    | EventAssignment | Assignment to amount-tracked `hasOnlySubstanceUnits=false` species: spec says math must match species quantity unit (§4.12.5) but does not specify the conversion procedure. pysbml multiplies the assigned concentration value by compartment size to obtain amount.                                                                                                                                                                     |
| D9  | `SPEC_SILENT`    | Event           | `rateOf` csymbol supported via `__rateOf_X__` sentinel and `substitute_rate_of()`. `rateOf` is an L3 package extension, not part of L3v2 Core. It is accepted silently rather than rejected or flagged. For amount-tracked species: `rateOf(S)` = `rate_amount/C - S_conc/C * dC_rate`.                                                                                                                                                   |
| D10 | `SPEC_EXTENSION` | Event / MathML  | `delay(x, d)` csymbol resolved in ANY expression (kinetic laws, derived rules, IAs) via time-shift substitution or `Piecewise` history for static variables. Spec §4.12.4 only defines Delay in the context of Events. True DDEs (dynamic variables) raise NotImplementedError.                                                                                                                                                           |
| D11 | `SPEC_EXTENSION` | Reaction        | Deferred QSS: when a fast reaction is inactive at t=0 (net flux = 0 at initial params), species stays as state variable. Event assignments are injected to snap species to QSS value when events activate the fast reaction. Entirely beyond spec.                                                                                                                                                                                        |
| D12 | `SPEC_SILENT`    | EventAssignment | `_handle_conc_boundary`: when a compartment event fires without assigning the boundary species, pysbml auto-adjusts `{k}_conc_new = {k}_conc_old * C_old / C_new` to conserve amount. Spec does not specify this conservation behavior.                                                                                                                                                                                                   |
| D13 | `SPEC_CONFLICT`  | Species         | `constant=true`, `hasOnlySubstanceUnits=false`, initialized via `initialConcentration` in a non-constant compartment: spec §4.6.4 states it is the **amount** that is held constant. pysbml stores the raw `initialConcentration` value as the constant parameter, effectively pinning the concentration instead of the amount. Bug is latent when `C(0)=1` (cases 01117, 01118, 01377), but would produce wrong results when `C(0) ≠ 1`. |

---


### Per-Divergence Counts (pysbml, v3 scanner)

| Divergence | Cases | Notes                                                     |
| ---------- | ----- | --------------------------------------------------------- |
| D1         | 65    | Species with no initial value                             |
| D2         | 12    | HOSU=true + BC=true                                       |
| D3         | 0     | fast=true (no L3v2 test uses it)                          |
| D4         | 833   | Compartment symbol in kinetic law                         |
| D5         | 19    | Rate rule + dynamic compartment chain rule                |
| D6         | 102   | AlgebraicRule present                                     |
| D7         | 1     | Constraint present                                        |
| D8         | 258   | EventAssignment to HOSU=false species                     |
| D9         | 62    | rateOf csymbol                                            |
| D10        | 52    | delay csymbol                                             |
| D11        | 0     | Deferred QSS (statically undetectable)                    |
| D12        | 1     | Compartment event without species assignment              |
| D13        | 3     | constant + HOSU=false + initialConc + dynamic compartment |
| none       | 652   | No divergences                                            |

---

## L3v2 Test Case Divergence

| Case  | Status         |
| ----- | -------------- |
| 00001 | D4             |
| 00002 | D4             |
| 00003 | D4             |
| 00004 | D4             |
| 00005 | D4             |
| 00006 | D4             |
| 00007 | D4             |
| 00008 | D4             |
| 00009 | D4             |
| 00010 | D4             |
| 00011 | D4             |
| 00012 | D4             |
| 00013 | D4             |
| 00014 | D4             |
| 00015 | D4             |
| 00016 | D4             |
| 00017 | D4             |
| 00018 | D4             |
| 00019 | D4             |
| 00020 | D4             |
| 00021 | D4             |
| 00022 | D4             |
| 00023 | D4             |
| 00024 | D4             |
| 00025 | D4             |
| 00026 | D4, D8         |
| 00027 | D4             |
| 00028 | none           |
| 00029 | none           |
| 00030 | D1             |
| 00031 | none           |
| 00032 | none           |
| 00033 | D4             |
| 00034 | D4             |
| 00035 | D4             |
| 00036 | D4             |
| 00037 | D4             |
| 00038 | D1, D4         |
| 00039 | D4, D6         |
| 00040 | D6             |
| 00041 | D4, D8         |
| 00042 | D4             |
| 00043 | D4             |
| 00044 | D4             |
| 00045 | D4             |
| 00046 | D4             |
| 00047 | D4             |
| 00048 | none           |
| 00049 | none           |
| 00050 | none           |
| 00051 | D4             |
| 00052 | D4             |
| 00053 | D4             |
| 00054 | D4             |
| 00055 | D4             |
| 00056 | D4             |
| 00057 | D4             |
| 00058 | D4             |
| 00060 | none           |
| 00061 | none           |
| 00062 | none           |
| 00063 | D4             |
| 00064 | D4             |
| 00065 | D4             |
| 00066 | D4             |
| 00067 | D4             |
| 00071 | D4, D8         |
| 00072 | D4, D8         |
| 00073 | D4, D8         |
| 00074 | D4, D8         |
| 00075 | D4             |
| 00076 | D4             |
| 00077 | D4             |
| 00078 | D1, D4         |
| 00079 | D1, D4         |
| 00080 | D4             |
| 00081 | D4             |
| 00082 | D4             |
| 00083 | D4             |
| 00084 | D4             |
| 00085 | D4             |
| 00086 | D4             |
| 00087 | D4             |
| 00088 | D4             |
| 00089 | D1, D4         |
| 00090 | D4             |
| 00091 | D4             |
| 00092 | D4             |
| 00093 | D1, D4         |
| 00094 | D4             |
| 00095 | D4             |
| 00096 | D4             |
| 00097 | none           |
| 00098 | D4             |
| 00099 | D4             |
| 00100 | none           |
| 00101 | D4             |
| 00102 | D4             |
| 00103 | none           |
| 00104 | D4             |
| 00105 | D4             |
| 00106 | D4             |
| 00107 | D4             |
| 00108 | D4             |
| 00109 | D4             |
| 00110 | D4             |
| 00111 | D4             |
| 00112 | D4             |
| 00113 | none           |
| 00114 | none           |
| 00115 | none           |
| 00116 | D4             |
| 00117 | D4             |
| 00118 | D4             |
| 00119 | D4             |
| 00120 | D4             |
| 00121 | D4             |
| 00122 | D4             |
| 00123 | D4             |
| 00124 | D4             |
| 00125 | D4             |
| 00126 | D4             |
| 00127 | D4             |
| 00128 | D4             |
| 00132 | D4             |
| 00133 | D4             |
| 00135 | D4             |
| 00136 | D4             |
| 00137 | D1, D4         |
| 00138 | D4             |
| 00139 | D4             |
| 00140 | D4             |
| 00141 | D4             |
| 00142 | D4             |
| 00143 | D4             |
| 00144 | D4             |
| 00145 | D4             |
| 00146 | D4             |
| 00147 | D4             |
| 00148 | D4             |
| 00149 | D4             |
| 00150 | D4             |
| 00151 | D1, D4         |
| 00152 | D1, D4         |
| 00153 | D1, D4         |
| 00154 | D1, D4         |
| 00155 | D1             |
| 00156 | D1             |
| 00157 | D1, D4         |
| 00158 | D1, D4         |
| 00159 | D1, D4         |
| 00160 | D1, D4         |
| 00161 | none           |
| 00162 | none           |
| 00163 | none           |
| 00164 | none           |
| 00165 | none           |
| 00166 | none           |
| 00167 | none           |
| 00168 | none           |
| 00169 | none           |
| 00170 | none           |
| 00171 | none           |
| 00172 | none           |
| 00173 | none           |
| 00174 | none           |
| 00175 | none           |
| 00176 | none           |
| 00177 | none           |
| 00178 | none           |
| 00179 | none           |
| 00180 | none           |
| 00181 | none           |
| 00182 | D6             |
| 00183 | none           |
| 00184 | D6             |
| 00185 | none           |
| 00186 | D4             |
| 00187 | D4             |
| 00188 | D4             |
| 00189 | D4             |
| 00190 | none           |
| 00191 | D4             |
| 00192 | D4             |
| 00193 | D4             |
| 00194 | D4             |
| 00195 | D4             |
| 00196 | none           |
| 00197 | none           |
| 00198 | D4             |
| 00199 | D4             |
| 00200 | D4             |
| 00201 | D4             |
| 00202 | D4             |
| 00203 | D4             |
| 00204 | D4             |
| 00205 | D4             |
| 00206 | D4             |
| 00207 | D4             |
| 00208 | none           |
| 00209 | none           |
| 00210 | none           |
| 00211 | D4             |
| 00212 | D4             |
| 00213 | D4             |
| 00214 | D4             |
| 00215 | D4             |
| 00216 | D4             |
| 00217 | D4             |
| 00218 | D4             |
| 00219 | D4             |
| 00220 | D4             |
| 00221 | D4             |
| 00222 | D4             |
| 00223 | none           |
| 00224 | none           |
| 00225 | none           |
| 00226 | D4             |
| 00227 | D4             |
| 00228 | D4             |
| 00229 | D4             |
| 00230 | D4             |
| 00231 | D4             |
| 00232 | D4             |
| 00233 | D4             |
| 00234 | D4             |
| 00235 | D4             |
| 00236 | D4             |
| 00237 | D4             |
| 00238 | none           |
| 00239 | none           |
| 00240 | none           |
| 00241 | none           |
| 00242 | none           |
| 00243 | none           |
| 00244 | none           |
| 00245 | none           |
| 00246 | none           |
| 00247 | none           |
| 00248 | D4             |
| 00249 | D4             |
| 00250 | D4             |
| 00251 | D4             |
| 00252 | D4             |
| 00253 | D4             |
| 00254 | D4             |
| 00255 | D4             |
| 00256 | D4             |
| 00257 | none           |
| 00258 | none           |
| 00259 | none           |
| 00260 | D4             |
| 00261 | D4             |
| 00262 | none           |
| 00263 | D4             |
| 00264 | D4             |
| 00265 | none           |
| 00266 | D4             |
| 00267 | D4             |
| 00268 | none           |
| 00269 | none           |
| 00270 | D4             |
| 00271 | none           |
| 00272 | D4             |
| 00273 | D4             |
| 00274 | D4             |
| 00275 | D4             |
| 00276 | D4             |
| 00277 | D4             |
| 00278 | D4             |
| 00279 | D4             |
| 00280 | D4             |
| 00281 | D4             |
| 00282 | D4             |
| 00283 | D4             |
| 00284 | D4             |
| 00285 | D4             |
| 00286 | D4             |
| 00287 | D4             |
| 00288 | D4             |
| 00289 | D4             |
| 00290 | D1, D4         |
| 00291 | D4             |
| 00292 | D1, D4         |
| 00293 | D4             |
| 00294 | D1, D4         |
| 00295 | D1             |
| 00296 | none           |
| 00297 | D1, D4         |
| 00298 | D1, D4         |
| 00299 | D4             |
| 00300 | D4             |
| 00301 | D4             |
| 00302 | D4             |
| 00303 | D1, D4         |
| 00304 | D1, D4         |
| 00305 | D1, D4         |
| 00306 | D4             |
| 00307 | D4             |
| 00308 | D4             |
| 00309 | D1, D4         |
| 00310 | D4             |
| 00311 | D4             |
| 00312 | D4             |
| 00313 | D4             |
| 00314 | D4             |
| 00315 | D4             |
| 00316 | D4             |
| 00317 | D4             |
| 00318 | D4             |
| 00319 | D4             |
| 00320 | D4             |
| 00321 | none           |
| 00322 | D4             |
| 00323 | D4             |
| 00324 | none           |
| 00325 | D4             |
| 00326 | D4             |
| 00327 | none           |
| 00328 | D4             |
| 00329 | D4             |
| 00330 | D4             |
| 00331 | none           |
| 00332 | none           |
| 00333 | none           |
| 00334 | D4             |
| 00335 | D4             |
| 00336 | D4             |
| 00337 | D4             |
| 00338 | D4             |
| 00339 | D4             |
| 00340 | D4             |
| 00341 | D4             |
| 00342 | D4             |
| 00343 | D4             |
| 00344 | D4             |
| 00345 | D4             |
| 00346 | D4             |
| 00347 | D4             |
| 00348 | D4, D8         |
| 00349 | D4, D8         |
| 00350 | D4, D8         |
| 00351 | D4, D8         |
| 00352 | D4, D8         |
| 00353 | D4, D8         |
| 00354 | D4, D8         |
| 00355 | D4, D8         |
| 00356 | D4, D8         |
| 00357 | D4, D8         |
| 00358 | D4, D8         |
| 00359 | D4, D8         |
| 00360 | D4, D8         |
| 00361 | D4, D8         |
| 00362 | D8             |
| 00363 | D4, D8         |
| 00364 | D4, D8         |
| 00365 | D8             |
| 00366 | D4, D8         |
| 00367 | D4, D8         |
| 00368 | D8             |
| 00369 | D4, D8         |
| 00370 | D4, D8         |
| 00371 | D4, D8         |
| 00372 | D4, D8         |
| 00373 | D4, D8         |
| 00374 | D4, D8         |
| 00375 | D4, D8         |
| 00376 | D4, D8         |
| 00377 | D4, D8         |
| 00378 | D4, D8         |
| 00379 | D4, D8         |
| 00380 | D4, D8         |
| 00381 | D4, D8         |
| 00382 | D4, D8         |
| 00383 | D4, D8         |
| 00384 | D4, D8         |
| 00385 | D4, D8         |
| 00386 | D4, D8         |
| 00387 | D4, D8         |
| 00389 | D4, D8         |
| 00390 | D4, D8         |
| 00392 | D4, D8         |
| 00393 | D4, D8         |
| 00395 | D4, D8         |
| 00396 | none           |
| 00397 | none           |
| 00398 | none           |
| 00399 | D4, D8         |
| 00400 | D4, D8         |
| 00401 | D4, D8         |
| 00402 | none           |
| 00403 | none           |
| 00404 | none           |
| 00405 | D4, D8         |
| 00406 | D4, D8         |
| 00407 | D4, D8         |
| 00408 | D4, D8         |
| 00409 | D4, D8         |
| 00410 | D4, D8         |
| 00411 | D4, D8         |
| 00412 | D4, D8         |
| 00413 | D4, D8         |
| 00414 | D4, D8         |
| 00415 | D4, D8         |
| 00416 | D4, D8         |
| 00417 | D4, D8         |
| 00418 | D4, D8         |
| 00419 | D8             |
| 00420 | D4, D8         |
| 00421 | D4, D8         |
| 00422 | D8             |
| 00423 | D4, D8         |
| 00424 | D4, D8         |
| 00425 | D8             |
| 00426 | D4, D8         |
| 00427 | D4, D8         |
| 00428 | D4, D8         |
| 00429 | D4, D8         |
| 00430 | D4, D8         |
| 00431 | D4, D8         |
| 00432 | D4, D8         |
| 00433 | D4, D8         |
| 00434 | D4, D8         |
| 00435 | D4, D8         |
| 00436 | D4, D8         |
| 00437 | D4, D8         |
| 00438 | D4, D8         |
| 00439 | D4, D8         |
| 00440 | D4, D8         |
| 00441 | D4, D8         |
| 00442 | D4, D8         |
| 00443 | D4, D8         |
| 00444 | D4, D8         |
| 00446 | D4, D8         |
| 00447 | D4, D8         |
| 00449 | D4, D8         |
| 00450 | D4, D8         |
| 00452 | D4, D8         |
| 00453 | none           |
| 00454 | none           |
| 00455 | none           |
| 00456 | D4, D8         |
| 00457 | D4, D8         |
| 00458 | D4, D8         |
| 00459 | none           |
| 00460 | none           |
| 00461 | none           |
| 00462 | D4             |
| 00463 | D4             |
| 00464 | D4             |
| 00465 | D4             |
| 00466 | D4             |
| 00467 | D4             |
| 00468 | D4             |
| 00469 | D4             |
| 00470 | D4             |
| 00471 | D4             |
| 00472 | D4             |
| 00473 | D4             |
| 00474 | D4             |
| 00475 | D4             |
| 00476 | D4             |
| 00477 | D4             |
| 00478 | D4             |
| 00479 | D4             |
| 00480 | D4             |
| 00481 | D4             |
| 00482 | D4             |
| 00483 | D4             |
| 00484 | D4             |
| 00485 | D4             |
| 00486 | D4             |
| 00487 | D4             |
| 00488 | D4             |
| 00489 | none           |
| 00490 | none           |
| 00491 | none           |
| 00492 | D4             |
| 00493 | D4             |
| 00494 | D4             |
| 00495 | D4             |
| 00496 | D4             |
| 00497 | D4             |
| 00498 | D4             |
| 00499 | D4             |
| 00500 | D4             |
| 00501 | D4             |
| 00502 | D4             |
| 00503 | D4             |
| 00504 | D4             |
| 00505 | D4             |
| 00506 | D4             |
| 00507 | D4             |
| 00508 | D4             |
| 00509 | D4             |
| 00510 | D4             |
| 00511 | D4             |
| 00512 | D4             |
| 00513 | D4             |
| 00514 | D4             |
| 00515 | D4             |
| 00522 | D4             |
| 00523 | D4             |
| 00524 | D4             |
| 00525 | D4             |
| 00526 | D4             |
| 00527 | D4             |
| 00528 | D4             |
| 00529 | D4             |
| 00530 | D4             |
| 00531 | D4, D6         |
| 00532 | D4, D6         |
| 00533 | D4, D6         |
| 00534 | D4, D6         |
| 00535 | D4, D6         |
| 00536 | D4, D6         |
| 00537 | D4, D6         |
| 00538 | D4, D6         |
| 00539 | D4, D6         |
| 00540 | D4, D6         |
| 00541 | D4, D6         |
| 00542 | D4, D6         |
| 00543 | D6             |
| 00544 | D4, D6         |
| 00545 | D4, D6         |
| 00546 | D6             |
| 00547 | D4, D6         |
| 00548 | D4, D6         |
| 00549 | D4, D6         |
| 00550 | D4, D6         |
| 00551 | D4, D6         |
| 00552 | D4, D6         |
| 00553 | D4, D6         |
| 00554 | D4, D6         |
| 00555 | D4, D6         |
| 00556 | D4, D6         |
| 00557 | D4, D6         |
| 00558 | D4, D6         |
| 00559 | D4, D6         |
| 00560 | D4, D6         |
| 00565 | D4, D6         |
| 00566 | D4, D6         |
| 00567 | D4, D6         |
| 00568 | D4, D6         |
| 00569 | D4, D6         |
| 00570 | D4, D6         |
| 00571 | D4, D6         |
| 00572 | D4, D6         |
| 00573 | D6             |
| 00574 | D6             |
| 00575 | D6             |
| 00576 | D6             |
| 00577 | D4             |
| 00578 | D4             |
| 00579 | D4             |
| 00580 | D4             |
| 00581 | D4             |
| 00582 | D4             |
| 00583 | D4             |
| 00584 | D4             |
| 00585 | D4             |
| 00586 | D4             |
| 00587 | D4             |
| 00588 | D4             |
| 00589 | D4             |
| 00590 | D4             |
| 00591 | D4             |
| 00592 | D4             |
| 00593 | D4             |
| 00594 | D4             |
| 00595 | D4             |
| 00596 | D4             |
| 00598 | D4             |
| 00599 | D4             |
| 00600 | D4             |
| 00601 | D4             |
| 00602 | D4             |
| 00603 | D4             |
| 00604 | D4             |
| 00605 | D4             |
| 00606 | D4             |
| 00607 | D1, D4         |
| 00608 | D1, D4         |
| 00611 | D4             |
| 00612 | D1, D4         |
| 00613 | D4, D6         |
| 00614 | D4, D6         |
| 00615 | D4, D6         |
| 00616 | D4             |
| 00617 | D4             |
| 00618 | D4             |
| 00619 | D1, D4, D8     |
| 00620 | D4, D8         |
| 00621 | D4, D8         |
| 00622 | D1, D4, D8     |
| 00623 | D4, D8         |
| 00624 | D4, D8         |
| 00625 | D4             |
| 00626 | D1, D4         |
| 00627 | D4             |
| 00628 | D4, D6         |
| 00629 | D4, D6         |
| 00630 | D4, D6         |
| 00631 | D4             |
| 00632 | D4             |
| 00633 | D4             |
| 00634 | D1, D4, D8     |
| 00635 | D4, D8         |
| 00636 | D4, D8         |
| 00637 | D1, D4, D8     |
| 00638 | D4, D8         |
| 00639 | D4, D8         |
| 00640 | D4             |
| 00641 | D4             |
| 00642 | D4             |
| 00643 | D4             |
| 00644 | D4             |
| 00645 | D4             |
| 00646 | D4, D8         |
| 00647 | D4, D8         |
| 00648 | D4, D8         |
| 00649 | D4, D8         |
| 00650 | D4, D8         |
| 00651 | D4, D8         |
| 00652 | D4, D8         |
| 00653 | D1, D4, D8     |
| 00654 | D4, D8         |
| 00655 | D4, D8         |
| 00656 | D1, D4, D8     |
| 00657 | D4, D8         |
| 00658 | D4, D6         |
| 00659 | D4, D6         |
| 00660 | D4, D6         |
| 00661 | D4, D6, D8     |
| 00662 | D4, D6, D8     |
| 00663 | D4, D6, D8     |
| 00664 | D4, D6, D8     |
| 00665 | D4, D6, D8     |
| 00666 | D4, D6, D8     |
| 00667 | D1, D4         |
| 00668 | D1, D4         |
| 00669 | D4             |
| 00670 | D4             |
| 00671 | D1, D4         |
| 00672 | D4             |
| 00673 | D4, D6         |
| 00674 | D4, D6         |
| 00675 | D4, D6         |
| 00676 | D4             |
| 00677 | D4             |
| 00678 | D4             |
| 00679 | D1, D4, D8     |
| 00680 | D4, D8         |
| 00681 | D4, D8         |
| 00682 | D1, D4, D8     |
| 00683 | D4, D8         |
| 00684 | D4, D8         |
| 00685 | D1, D4         |
| 00686 | D4             |
| 00687 | D4, D6         |
| 00688 | D4             |
| 00689 | D1, D4, D8     |
| 00690 | D1, D4, D8     |
| 00691 | D1, D4         |
| 00692 | D1, D4         |
| 00693 | D4             |
| 00694 | D4             |
| 00695 | D4, D6         |
| 00696 | D4, D6         |
| 00697 | D4             |
| 00698 | D4             |
| 00699 | D1, D4, D8     |
| 00700 | D1, D4, D8     |
| 00701 | D1, D4, D8     |
| 00702 | D1, D4, D8     |
| 00703 | D1, D4         |
| 00704 | D1, D4         |
| 00705 | D4, D6         |
| 00706 | D4             |
| 00707 | D4, D8         |
| 00708 | D4, D8         |
| 00709 | D4             |
| 00710 | D4             |
| 00711 | D4             |
| 00712 | D4             |
| 00713 | D4             |
| 00714 | D4             |
| 00715 | D1, D4         |
| 00716 | D4             |
| 00717 | D4             |
| 00718 | D4             |
| 00719 | D4             |
| 00720 | D4             |
| 00721 | D1, D4         |
| 00722 | D4             |
| 00723 | D4, D8         |
| 00724 | D4, D8         |
| 00732 | D4             |
| 00733 | D4             |
| 00734 | D1, D4         |
| 00735 | D4             |
| 00736 | D4, D8         |
| 00737 | D4, D8         |
| 00738 | D1, D4         |
| 00739 | D4             |
| 00740 | D4             |
| 00741 | D4             |
| 00742 | D4             |
| 00743 | D4, D8         |
| 00744 | D4, D8         |
| 00745 | D4, D8         |
| 00746 | D4, D8         |
| 00747 | D4, D8         |
| 00748 | D4, D8         |
| 00749 | D4, D8         |
| 00750 | D4, D8         |
| 00751 | D4, D8         |
| 00752 | D4, D8         |
| 00753 | D4, D8         |
| 00754 | D4, D8         |
| 00755 | D4, D8         |
| 00756 | D4, D8         |
| 00757 | D4, D8         |
| 00758 | D4, D8         |
| 00759 | D4, D8         |
| 00760 | D4, D6, D8     |
| 00761 | D4, D6, D8     |
| 00762 | D4, D6, D8     |
| 00763 | D4, D8         |
| 00764 | D4, D8         |
| 00765 | D4, D8         |
| 00766 | D4, D8         |
| 00767 | D4, D8         |
| 00768 | D4, D8         |
| 00769 | D4, D8         |
| 00770 | D4, D8         |
| 00771 | D4, D8         |
| 00772 | D4, D8         |
| 00773 | D4, D8         |
| 00774 | D4, D8         |
| 00775 | D4, D8         |
| 00776 | D4, D8         |
| 00777 | D1, D4, D6, D8 |
| 00778 | D4, D6, D8     |
| 00779 | D4, D6, D8     |
| 00780 | D4, D6, D8     |
| 00781 | D4             |
| 00782 | D4             |
| 00783 | D4             |
| 00784 | D4             |
| 00785 | D4             |
| 00786 | D4             |
| 00787 | D4             |
| 00788 | D4             |
| 00789 | D4, D8         |
| 00790 | D4, D8         |
| 00791 | D4, D8         |
| 00792 | D4             |
| 00793 | D4             |
| 00794 | D4             |
| 00795 | D4             |
| 00796 | D4             |
| 00797 | D4             |
| 00798 | D4             |
| 00799 | D4             |
| 00800 | D4             |
| 00801 | D4             |
| 00802 | D4             |
| 00803 | D4             |
| 00804 | D4             |
| 00805 | D4             |
| 00806 | D4             |
| 00807 | D4             |
| 00808 | D4             |
| 00809 | D4             |
| 00810 | D4             |
| 00811 | D4             |
| 00812 | D4             |
| 00813 | D4             |
| 00814 | D4             |
| 00815 | D4             |
| 00816 | D4             |
| 00817 | D4             |
| 00818 | D4             |
| 00819 | D4             |
| 00820 | D4             |
| 00821 | D4             |
| 00822 | D4             |
| 00823 | D4             |
| 00824 | D4             |
| 00825 | D4             |
| 00826 | D4             |
| 00830 | D4             |
| 00831 | D4             |
| 00832 | D4             |
| 00833 | D4             |
| 00834 | D4             |
| 00835 | D4             |
| 00836 | D4             |
| 00837 | D4             |
| 00838 | D4             |
| 00839 | D1, D4         |
| 00840 | D4             |
| 00841 | D4             |
| 00842 | D4             |
| 00843 | D4             |
| 00844 | D4, D6         |
| 00845 | D4, D8         |
| 00846 | D4, D8         |
| 00847 | D4, D8         |
| 00848 | D4, D8         |
| 00849 | D4, D8         |
| 00850 | D4, D8         |
| 00851 | D4             |
| 00852 | D4             |
| 00853 | D4             |
| 00854 | D4             |
| 00855 | D4             |
| 00856 | D4             |
| 00857 | D4             |
| 00858 | D4             |
| 00859 | D4             |
| 00860 | D4             |
| 00861 | D4             |
| 00862 | D4             |
| 00863 | D4             |
| 00864 | D4             |
| 00865 | D4             |
| 00866 | D4             |
| 00867 | D4             |
| 00868 | D4             |
| 00869 | D4             |
| 00876 | D4, D6         |
| 00877 | D4             |
| 00878 | D4             |
| 00879 | D4             |
| 00880 | D1, D4         |
| 00881 | D4             |
| 00882 | D4             |
| 00883 | D4, D8         |
| 00884 | D4, D8         |
| 00885 | D4, D8         |
| 00886 | D4, D8         |
| 00887 | D4, D8         |
| 00888 | D4             |
| 00889 | D4             |
| 00890 | D4             |
| 00891 | none           |
| 00892 | none           |
| 00893 | none           |
| 00894 | D4             |
| 00895 | D4             |
| 00896 | D4             |
| 00897 | D4             |
| 00901 | none           |
| 00902 | none           |
| 00903 | none           |
| 00904 | none           |
| 00905 | none           |
| 00906 | none           |
| 00907 | none           |
| 00908 | none           |
| 00909 | none           |
| 00910 | none           |
| 00911 | none           |
| 00912 | none           |
| 00913 | none           |
| 00914 | none           |
| 00915 | none           |
| 00916 | none           |
| 00917 | none           |
| 00918 | none           |
| 00919 | none           |
| 00920 | none           |
| 00921 | none           |
| 00922 | none           |
| 00923 | none           |
| 00924 | none           |
| 00925 | none           |
| 00926 | none           |
| 00927 | none           |
| 00928 | D4, D8         |
| 00929 | D4, D8         |
| 00930 | D8             |
| 00931 | D8             |
| 00932 | D4, D8         |
| 00933 | D4, D8         |
| 00934 | D8             |
| 00935 | D8             |
| 00936 | D1, D8         |
| 00937 | D10            |
| 00938 | D10            |
| 00939 | D10            |
| 00940 | D10            |
| 00941 | D10            |
| 00942 | D10            |
| 00943 | D10            |
| 00944 | D4             |
| 00945 | D4             |
| 00946 | D4             |
| 00947 | D4             |
| 00948 | D4             |
| 00949 | none           |
| 00950 | none           |
| 00951 | none           |
| 00952 | none           |
| 00953 | none           |
| 00954 | none           |
| 00955 | none           |
| 00956 | none           |
| 00957 | none           |
| 00958 | none           |
| 00959 | none           |
| 00960 | none           |
| 00961 | none           |
| 00962 | none           |
| 00963 | none           |
| 00964 | none           |
| 00965 | none           |
| 00966 | none           |
| 00967 | none           |
| 00969 | none           |
| 00970 | none           |
| 00971 | none           |
| 00972 | none           |
| 00974 | none           |
| 00975 | none           |
| 00976 | none           |
| 00977 | none           |
| 00978 | none           |
| 00979 | none           |
| 00980 | none           |
| 00981 | D10            |
| 00982 | D10            |
| 00983 | D10, D6        |
| 00984 | D10            |
| 00985 | D10            |
| 00995 | none           |
| 00996 | none           |
| 00997 | none           |
| 00998 | none           |
| 00999 | none           |
| 01000 | D2             |
| 01001 | none           |
| 01002 | none           |
| 01003 | none           |
| 01004 | none           |
| 01005 | none           |
| 01006 | none           |
| 01007 | none           |
| 01008 | none           |
| 01009 | none           |
| 01010 | none           |
| 01011 | none           |
| 01012 | none           |
| 01013 | none           |
| 01014 | D1             |
| 01015 | none           |
| 01016 | none           |
| 01017 | none           |
| 01018 | D4             |
| 01019 | D4             |
| 01020 | D4             |
| 01021 | D4             |
| 01022 | D4             |
| 01023 | D4             |
| 01024 | D4             |
| 01025 | D4             |
| 01026 | D4             |
| 01030 | D4             |
| 01031 | D4             |
| 01032 | D4             |
| 01033 | D4             |
| 01034 | D4             |
| 01035 | D4             |
| 01036 | D4             |
| 01037 | D4             |
| 01038 | D4             |
| 01039 | D1, D4         |
| 01040 | D4             |
| 01041 | D4             |
| 01042 | D4             |
| 01043 | D4             |
| 01044 | D4, D6         |
| 01045 | D4, D8         |
| 01046 | D4, D8         |
| 01047 | D4, D8         |
| 01048 | D4, D8         |
| 01049 | D4, D8         |
| 01050 | D4, D8         |
| 01054 | D4, D6         |
| 01055 | D4             |
| 01056 | D4             |
| 01057 | D4             |
| 01058 | D4             |
| 01059 | D4             |
| 01060 | D4             |
| 01061 | D4             |
| 01062 | D4             |
| 01063 | D4             |
| 01064 | D4             |
| 01065 | D4             |
| 01066 | D4             |
| 01067 | D4             |
| 01068 | D4             |
| 01069 | D4             |
| 01070 | D4             |
| 01071 | D4, D8         |
| 01072 | D4, D8         |
| 01073 | D4, D8         |
| 01074 | D4, D8         |
| 01075 | D4, D8         |
| 01076 | D4, D8         |
| 01077 | D4             |
| 01078 | D4             |
| 01079 | D4             |
| 01080 | D4             |
| 01081 | D4             |
| 01082 | D4             |
| 01083 | D4, D6         |
| 01084 | D4, D6         |
| 01085 | D4, D6         |
| 01086 | D4, D6         |
| 01087 | D4             |
| 01088 | D1, D4         |
| 01089 | D4             |
| 01090 | D4             |
| 01091 | D4             |
| 01092 | D1, D4         |
| 01093 | D4             |
| 01094 | D4, D8         |
| 01095 | D4, D8         |
| 01096 | D4             |
| 01097 | D4             |
| 01098 | D4             |
| 01099 | D4             |
| 01100 | D4             |
| 01101 | D4             |
| 01102 | none           |
| 01103 | none           |
| 01104 | none           |
| 01105 | none           |
| 01106 | none           |
| 01107 | none           |
| 01108 | D6             |
| 01109 | none           |
| 01110 | D4             |
| 01111 | D4             |
| 01112 | none           |
| 01113 | none           |
| 01114 | none           |
| 01115 | none           |
| 01116 | none           |
| 01117 | D13            |
| 01118 | D13            |
| 01119 | none           |
| 01120 | none           |
| 01121 | D2             |
| 01122 | none           |
| 01123 | D2             |
| 01124 | none           |
| 01125 | none           |
| 01126 | none           |
| 01127 | none           |
| 01128 | none           |
| 01129 | none           |
| 01130 | none           |
| 01131 | none           |
| 01132 | none           |
| 01133 | none           |
| 01134 | none           |
| 01135 | none           |
| 01136 | none           |
| 01137 | none           |
| 01138 | none           |
| 01139 | none           |
| 01140 | none           |
| 01141 | none           |
| 01142 | D10            |
| 01143 | none           |
| 01144 | none           |
| 01145 | none           |
| 01146 | none           |
| 01147 | none           |
| 01148 | none           |
| 01149 | none           |
| 01150 | none           |
| 01151 | none           |
| 01152 | none           |
| 01153 | none           |
| 01154 | none           |
| 01155 | none           |
| 01156 | none           |
| 01157 | none           |
| 01158 | none           |
| 01159 | none           |
| 01160 | none           |
| 01161 | none           |
| 01162 | none           |
| 01163 | none           |
| 01164 | none           |
| 01165 | none           |
| 01166 | none           |
| 01167 | none           |
| 01168 | none           |
| 01169 | none           |
| 01170 | none           |
| 01171 | none           |
| 01172 | none           |
| 01173 | D10            |
| 01174 | D10            |
| 01175 | none           |
| 01176 | D10            |
| 01177 | none           |
| 01178 | none           |
| 01179 | none           |
| 01180 | none           |
| 01181 | none           |
| 01182 | none           |
| 01183 | none           |
| 01184 | none           |
| 01185 | none           |
| 01186 | none           |
| 01187 | none           |
| 01188 | none           |
| 01189 | none           |
| 01190 | none           |
| 01191 | none           |
| 01192 | none           |
| 01193 | none           |
| 01194 | none           |
| 01195 | none           |
| 01196 | none           |
| 01197 | none           |
| 01198 | D5             |
| 01199 | none           |
| 01200 | none           |
| 01201 | none           |
| 01202 | none           |
| 01203 | none           |
| 01204 | none           |
| 01205 | none           |
| 01206 | none           |
| 01207 | D2             |
| 01208 | D5             |
| 01209 | none           |
| 01210 | none           |
| 01211 | none           |
| 01212 | none           |
| 01213 | none           |
| 01214 | none           |
| 01215 | none           |
| 01216 | none           |
| 01217 | none           |
| 01218 | none           |
| 01219 | D2             |
| 01220 | D2             |
| 01221 | D2             |
| 01222 | D2             |
| 01223 | D2             |
| 01224 | none           |
| 01225 | none           |
| 01226 | none           |
| 01227 | none           |
| 01228 | none           |
| 01229 | none           |
| 01230 | none           |
| 01231 | none           |
| 01232 | none           |
| 01233 | none           |
| 01234 | none           |
| 01235 | none           |
| 01236 | none           |
| 01237 | none           |
| 01238 | none           |
| 01239 | none           |
| 01240 | none           |
| 01241 | none           |
| 01242 | none           |
| 01243 | none           |
| 01244 | D6             |
| 01245 | none           |
| 01246 | none           |
| 01247 | D7             |
| 01248 | D9             |
| 01249 | D9             |
| 01250 | D9             |
| 01251 | D9             |
| 01252 | D9             |
| 01253 | D9             |
| 01254 | D9             |
| 01255 | D9             |
| 01256 | D9             |
| 01257 | D9             |
| 01258 | D9             |
| 01259 | D9             |
| 01260 | D9             |
| 01261 | D9             |
| 01262 | D9             |
| 01263 | D9             |
| 01264 | D9             |
| 01265 | D9             |
| 01266 | D9             |
| 01267 | D9             |
| 01268 | D9             |
| 01269 | D9             |
| 01270 | D9             |
| 01271 | none           |
| 01272 | none           |
| 01273 | none           |
| 01274 | none           |
| 01275 | none           |
| 01276 | none           |
| 01277 | none           |
| 01278 | none           |
| 01279 | none           |
| 01280 | none           |
| 01281 | none           |
| 01282 | none           |
| 01283 | none           |
| 01284 | none           |
| 01285 | none           |
| 01286 | none           |
| 01287 | none           |
| 01288 | none           |
| 01289 | none           |
| 01290 | none           |
| 01291 | none           |
| 01292 | D6             |
| 01293 | D9             |
| 01294 | D9             |
| 01295 | D9             |
| 01296 | D9             |
| 01297 | D9             |
| 01298 | D9             |
| 01299 | D9             |
| 01300 | none           |
| 01301 | none           |
| 01302 | none           |
| 01303 | none           |
| 01304 | none           |
| 01305 | none           |
| 01306 | none           |
| 01307 | none           |
| 01308 | none           |
| 01309 | D2             |
| 01310 | none           |
| 01311 | none           |
| 01312 | none           |
| 01313 | none           |
| 01314 | none           |
| 01315 | none           |
| 01316 | none           |
| 01317 | none           |
| 01318 | D10            |
| 01319 | D10            |
| 01320 | D10            |
| 01321 | D9             |
| 01322 | D9             |
| 01323 | none           |
| 01324 | none           |
| 01325 | none           |
| 01326 | none           |
| 01327 | none           |
| 01328 | none           |
| 01329 | none           |
| 01330 | none           |
| 01331 | none           |
| 01332 | none           |
| 01333 | none           |
| 01334 | none           |
| 01335 | none           |
| 01336 | none           |
| 01337 | none           |
| 01338 | none           |
| 01339 | none           |
| 01340 | none           |
| 01341 | none           |
| 01342 | none           |
| 01343 | none           |
| 01344 | none           |
| 01345 | none           |
| 01346 | none           |
| 01347 | none           |
| 01348 | none           |
| 01349 | none           |
| 01350 | none           |
| 01351 | none           |
| 01352 | none           |
| 01353 | none           |
| 01354 | none           |
| 01355 | none           |
| 01356 | none           |
| 01357 | none           |
| 01358 | none           |
| 01359 | none           |
| 01360 | none           |
| 01361 | none           |
| 01362 | none           |
| 01363 | none           |
| 01364 | none           |
| 01365 | none           |
| 01366 | none           |
| 01367 | none           |
| 01368 | none           |
| 01369 | none           |
| 01370 | none           |
| 01371 | none           |
| 01372 | none           |
| 01373 | none           |
| 01374 | none           |
| 01375 | none           |
| 01376 | none           |
| 01377 | D13            |
| 01378 | none           |
| 01379 | none           |
| 01380 | none           |
| 01381 | none           |
| 01382 | none           |
| 01383 | none           |
| 01384 | none           |
| 01385 | none           |
| 01386 | none           |
| 01387 | none           |
| 01388 | none           |
| 01389 | none           |
| 01390 | none           |
| 01391 | none           |
| 01392 | none           |
| 01393 | none           |
| 01394 | none           |
| 01395 | none           |
| 01400 | D10, D9        |
| 01401 | D10, D9        |
| 01402 | D9             |
| 01403 | D10, D9        |
| 01404 | D10            |
| 01405 | D9             |
| 01406 | D10, D9        |
| 01407 | D10            |
| 01408 | D9             |
| 01409 | D10, D9        |
| 01410 | D10            |
| 01411 | D10            |
| 01412 | D10            |
| 01413 | D10            |
| 01414 | D10            |
| 01415 | D10            |
| 01416 | D10            |
| 01417 | D10            |
| 01418 | D10            |
| 01419 | D10            |
| 01420 | none           |
| 01421 | none           |
| 01422 | none           |
| 01423 | none           |
| 01424 | none           |
| 01425 | none           |
| 01426 | none           |
| 01427 | none           |
| 01428 | none           |
| 01429 | none           |
| 01430 | none           |
| 01431 | none           |
| 01432 | none           |
| 01433 | none           |
| 01434 | none           |
| 01436 | none           |
| 01438 | none           |
| 01440 | none           |
| 01442 | none           |
| 01444 | none           |
| 01445 | none           |
| 01446 | none           |
| 01447 | none           |
| 01448 | none           |
| 01449 | none           |
| 01450 | none           |
| 01451 | none           |
| 01452 | none           |
| 01453 | none           |
| 01454 | D10, D5        |
| 01455 | D9             |
| 01456 | D9             |
| 01457 | D9             |
| 01458 | D9             |
| 01459 | D9             |
| 01460 | D9             |
| 01461 | D9             |
| 01462 | D5, D9         |
| 01463 | D2, D9         |
| 01464 | none           |
| 01465 | none           |
| 01466 | none           |
| 01467 | none           |
| 01468 | none           |
| 01469 | none           |
| 01470 | none           |
| 01471 | none           |
| 01472 | none           |
| 01473 | none           |
| 01474 | none           |
| 01475 | none           |
| 01476 | none           |
| 01477 | none           |
| 01478 | D5             |
| 01479 | D6             |
| 01480 | D10            |
| 01482 | D6, D9         |
| 01483 | D6, D9         |
| 01484 | D6             |
| 01485 | none           |
| 01486 | none           |
| 01487 | none           |
| 01488 | none           |
| 01489 | none           |
| 01490 | none           |
| 01491 | none           |
| 01492 | none           |
| 01493 | none           |
| 01494 | none           |
| 01495 | none           |
| 01496 | none           |
| 01497 | none           |
| 01498 | D5             |
| 01499 | D6             |
| 01500 | D6             |
| 01501 | D6             |
| 01502 | D6             |
| 01503 | D6             |
| 01504 | D5, D8         |
| 01505 | D12, D5        |
| 01506 | D5, D8         |
| 01507 | D5, D8         |
| 01508 | D5, D8         |
| 01509 | D5, D8         |
| 01510 | D5, D8         |
| 01511 | D5, D8         |
| 01512 | D5             |
| 01513 | D5             |
| 01514 | D5             |
| 01515 | none           |
| 01516 | none           |
| 01518 | D10            |
| 01519 | D10            |
| 01520 | D10            |
| 01521 | D10            |
| 01522 | D10            |
| 01523 | D10            |
| 01524 | D10            |
| 01525 | D9             |
| 01526 | D9             |
| 01527 | D9             |
| 01528 | D9             |
| 01529 | D9             |
| 01530 | none           |
| 01531 | none           |
| 01532 | none           |
| 01533 | none           |
| 01534 | D10            |
| 01535 | D10            |
| 01536 | D10            |
| 01537 | D10            |
| 01538 | D10            |
| 01540 | D9             |
| 01541 | D9             |
| 01542 | D9             |
| 01543 | D9             |
| 01552 | none           |
| 01553 | none           |
| 01554 | none           |
| 01555 | none           |
| 01556 | none           |
| 01557 | none           |
| 01561 | none           |
| 01563 | none           |
| 01564 | none           |
| 01566 | none           |
| 01574 | none           |
| 01575 | D6             |
| 01576 | D6             |
| 01577 | D6             |
| 01578 | D6             |
| 01579 | D6             |
| 01580 | none           |
| 01582 | none           |
| 01583 | none           |
| 01584 | none           |
| 01586 | none           |
| 01588 | none           |
| 01589 | D6             |
| 01590 | none           |
| 01591 | none           |
| 01592 | D10            |
| 01593 | D10            |
| 01594 | none           |
| 01595 | none           |
| 01596 | none           |
| 01597 | none           |
| 01598 | none           |
| 01599 | none           |
| 01600 | none           |
| 01601 | none           |
| 01602 | none           |
| 01603 | none           |
| 01604 | none           |
| 01605 | none           |
| 01606 | none           |
| 01607 | none           |
| 01608 | none           |
| 01609 | none           |
| 01610 | none           |
| 01611 | none           |
| 01612 | none           |
| 01613 | none           |
| 01614 | none           |
| 01615 | none           |
| 01616 | none           |
| 01617 | none           |
| 01618 | none           |
| 01619 | none           |
| 01620 | none           |
| 01621 | none           |
| 01622 | none           |
| 01623 | none           |
| 01624 | none           |
| 01625 | none           |
| 01626 | none           |
| 01627 | D5             |
| 01628 | none           |
| 01629 | none           |
| 01630 | none           |
| 01631 | none           |
| 01633 | none           |
| 01635 | none           |
| 01641 | none           |
| 01642 | none           |
| 01643 | none           |
| 01644 | none           |
| 01645 | none           |
| 01646 | none           |
| 01647 | none           |
| 01648 | none           |
| 01649 | none           |
| 01650 | none           |
| 01651 | none           |
| 01652 | none           |
| 01653 | none           |
| 01654 | none           |
| 01655 | none           |
| 01657 | none           |
| 01658 | none           |
| 01659 | none           |
| 01660 | none           |
| 01661 | none           |
| 01662 | none           |
| 01663 | none           |
| 01664 | none           |
| 01665 | none           |
| 01666 | none           |
| 01667 | none           |
| 01668 | none           |
| 01669 | D8             |
| 01670 | D8             |
| 01671 | D8             |
| 01672 | D8             |
| 01673 | D8             |
| 01674 | D8             |
| 01675 | D8             |
| 01676 | D8             |
| 01677 | D8             |
| 01678 | D8             |
| 01679 | D8             |
| 01680 | D8             |
| 01681 | D8             |
| 01682 | D8             |
| 01683 | D8             |
| 01684 | D8             |
| 01685 | D8             |
| 01686 | D8             |
| 01687 | D8             |
| 01688 | D8             |
| 01689 | D8             |
| 01690 | D8             |
| 01691 | D8             |
| 01692 | D8             |
| 01693 | none           |
| 01694 | none           |
| 01695 | none           |
| 01696 | none           |
| 01697 | none           |
| 01698 | none           |
| 01699 | none           |
| 01700 | none           |
| 01701 | none           |
| 01702 | none           |
| 01703 | none           |
| 01704 | none           |
| 01705 | D2             |
| 01706 | none           |
| 01707 | none           |
| 01708 | none           |
| 01709 | none           |
| 01710 | none           |
| 01711 | none           |
| 01712 | none           |
| 01713 | none           |
| 01714 | none           |
| 01715 | none           |
| 01716 | none           |
| 01717 | none           |
| 01718 | none           |
| 01719 | none           |
| 01720 | none           |
| 01721 | none           |
| 01722 | none           |
| 01723 | none           |
| 01724 | none           |
| 01725 | none           |
| 01726 | none           |
| 01727 | none           |
| 01728 | none           |
| 01729 | none           |
| 01730 | none           |
| 01731 | none           |
| 01732 | none           |
| 01733 | none           |
| 01734 | none           |
| 01735 | none           |
| 01736 | none           |
| 01737 | none           |
| 01738 | none           |
| 01739 | none           |
| 01740 | none           |
| 01741 | none           |
| 01742 | none           |
| 01744 | none           |
| 01746 | none           |
| 01748 | none           |
| 01754 | none           |
| 01755 | none           |
| 01756 | none           |
| 01757 | none           |
| 01758 | none           |
| 01759 | none           |
| 01760 | none           |
| 01761 | none           |
| 01762 | none           |
| 01763 | none           |
| 01766 | none           |
| 01767 | none           |
| 01768 | none           |
| 01769 | none           |
| 01770 | none           |
| 01771 | none           |
| 01772 | none           |
| 01773 | none           |
| 01774 | none           |
| 01775 | none           |
| 01776 | none           |
| 01777 | none           |
| 01778 | none           |
| 01779 | D8             |
| 01780 | D8             |
| 01781 | none           |
| 01782 | none           |
| 01783 | none           |
| 01799 | none           |
| 01800 | none           |
| 01801 | none           |
| 01802 | none           |
| 01803 | none           |
| 01804 | none           |
| 01805 | none           |
| 01806 | none           |
| 01807 | none           |
| 01808 | none           |
| 01809 | none           |
| 01810 | none           |
| 01811 | none           |
| 01812 | none           |
| 01813 | none           |
| 01814 | none           |
| 01815 | none           |
| 01816 | none           |
| 01817 | none           |
| 01818 | none           |
| 01819 | none           |
| 01820 | none           |
| 01821 | none           |
| 01822 | D9             |
| 01823 | D5, D9         |
