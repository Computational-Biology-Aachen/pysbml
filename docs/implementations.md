# SBML Implementation Comparison

Tracks how each tool transforms SBML L3v2 into an executable ODE system.
Organized by SBML element; each section has a `### LibraryName` subsection per tool.
Add a new `### LibraryName` block inside each element when comparing more tools.

**Spec:** SBML Level 3 Version 2 Core Release 2 (29 March 2019)
**Divergence flags:** See [`divergences.md`](divergences.md) for flag definitions and the full test-case table.

---

## Feature Matrix

| Feature                         | pysbml                                             | roadrunner                                           |
| ------------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| Internal species storage        | Dual: explicit `{k}_amount` + `{k}_conc` variables | Amount-canonical; concentration derived at read time |
| HOSU effect                     | Selects one of ~8 `_handle_*` dispatch paths       | Affects load/store conversion only                   |
| Algebraic rules                 | `sympy.solve` → assignment rule (D6)               | **Throws** (RR-A)                                    |
| Conservation law reduction      | None                                               | Optional L0-matrix moiety analysis (RR-B)            |
| Kinetic law compartment factor  | D4 auto-strip heuristic                            | None needed (HOSU loading cancels compartment)       |
| EventAssignment to HOSU=false   | Multiply by compartment (D8)                       | Multiply by compartment at store                     |
| Rate rule + dynamic compartment | Product rule correction (D5)                       | Same product rule                                    |
| rateOf csymbol                  | Sentinel + sympy substitution (D9)                 | Native LLVM IR codegen with quotient-rule            |
| delay csymbol                   | Time-shift approximation (D10)                     | **Throws** (RR-C)                                    |
| No initialAmount/initialConc    | 0.0 + amount-vs-conc heuristic (D1)                | 0.0 + `LOG_WARNING`                                  |
| Constraints                     | Silently ignored (D7)                              | Unknown                                              |
| fast=true reaction              | QSS reduction + deferred events (D3/D11)           | Unknown                                              |
| Conversion factors              | Applied to stoichiometry                           | EvalConversionFactorCodeGen                          |
| Dynamic stoichiometry           | SpeciesRef → parameter                             | EvalVolatileStoichCodeGen                            |

### Library Architectures

**pysbml** — `src/pysbml/` — parse layer (`parse/`), transform layer (`transform/`)
SBML → sympy ODE/algebraic system via two-stage pipeline (parse: libsbml → dataclasses; transform: dataclasses → mxlpy Model).

**roadrunner** — `ref/roadrunner/source/` (LLVM backend)
JIT-compiles SBML to native machine code via LLVM IR. Amount-canonical species storage; concentration derived at read/write boundaries.

---

## 1. FunctionDefinition (§4.3)

### Spec

- Lambda expression defining a reusable mathematical function.
- Arguments are positional; body is a MathML expression.
- Cannot be recursive (no cycles in function call graph).
- Identifiers of FunctionDefinitions may appear as function calls in MathML `<apply>` elements.

### pysbml

`convert_functions` in `transform/__init__.py`:

```
for each FunctionDefinition f with args (a1, ..., an) and body B:
    ctx.functions[f.id] = sympy.Lambda((a1, ..., an), parse_mathml(B))

# During kinetic law / rule parsing, function calls are inlined via substitution.
```

### roadrunner

Functions inlined at JIT compile time. No divergences noted.

### Divergences

None.

---

## 2. UnitDefinition (§4.4)

### Spec

- Named unit composed of base SI units with `kind`, `exponent`, `scale`, `multiplier`.
- Built-in reserved unit names (`substance`, `volume`, `area`, `length`, `time`) override model defaults when defined.
- SBML does not mandate unit enforcement — units are annotation.

### pysbml

`convert_units` in `transform/__init__.py`:

- Maps known SBML unit kinds to sympy unit objects.
- Determines `substance_unit` for the model (used for species scaling).
- Does **not** perform runtime unit enforcement.

### roadrunner

Units treated as annotation only. No divergences noted.

### Divergences

None relevant to ODE generation.

---

## 3. Compartment (§4.5)

### Spec

- `spatialDimensions`: 0, 1, 2, 3, or unset.
- `size`: initial value; if absent and no InitialAssignment, value is unknown.
- `constant="true"`: size never changes.
- `constant="false"`: size may change via RateRule, AssignmentRule, or EventAssignment.

### pysbml

`convert_compartments` in `transform/__init__.py`:

```
for each Compartment c:
    if c.is_constant:
        tmodel.parameters[c.id] = Parameter(value=c.size)
    elif c.id in rate_rules:
        tmodel.variables[c.id] = Variable(value=c.size)
        # rate rule becomes fake reaction "d{c.id}"
    else:
        tmodel.variables[c.id] = Variable(value=c.size)
        # governed by assignment rule or event assignment
```

### roadrunner

Constant compartments optimized at JIT compile time. Non-constant compartments tracked in model data. No divergences noted.

### Divergences

None observed.

---

## 4. Species (§4.6) — CRITICAL

### Spec

| Attribute               | Required | Default | Meaning                                                                        |
| ----------------------- | -------- | ------- | ------------------------------------------------------------------------------ |
| `compartment`           | yes      | —       | Enclosing compartment                                                          |
| `initialAmount`         | no       | —       | Initial amount (substance units)                                               |
| `initialConcentration`  | no       | —       | Initial concentration (amount/volume)                                          |
| `hasOnlySubstanceUnits` | yes      | false   | `true` → identifier = amount in formulas; `false` → identifier = concentration |
| `boundaryCondition`     | yes      | false   | `true` → reactions do not change this species                                  |
| `constant`              | yes      | false   | `true` → value never changes                                                   |
| `conversionFactor`      | no       | —       | Parameter id; scales stoichiometric contribution                               |

Key spec rules:
- At most one of `initialAmount` / `initialConcentration` should be set. If neither is set, initial value is "unknown or from an external source" (§4.6.4).
- When `constant="false"` and `boundaryCondition="false"`, reactions affect the species amount.
- The ODE describes rate of change of **amount** (not concentration): `dn_S/dt = convFactor_S * Σ_j(stoich_{S,Rj} · v_{Rj})`
- Rate rules on species describe `d(quantity)/dt` where quantity is amount if `hasOnlySubstanceUnits=true`, else concentration.
- For `constant="true"` + `hasOnlySubstanceUnits=false` + `initialConcentration` in a non-constant compartment: spec §4.6.4 states it is the **amount** that is held constant.

### pysbml

`_transform_species` in `transform/__init__.py`:

```python
def transform_species(k, species, pmodel, tmodel):

    # STEP 1: compartment validity
    # compartment_is_valid(pmodel, species): True if compartment exists AND
    #   (size != 0 and not nan) OR in assignment_rules OR in initial_assignments
    #   OR appears in algebraic rules.
    # variable_is_constant(k, pmodel): True if is_constant=True OR
    #   (has_boundary_condition=True AND no rate_rule AND no event_assignments targeting k).
    compartment = species.compartment
    if not compartment_is_valid(pmodel, species):
        if variable_is_constant(k, pmodel):
            _handle_constant_variable(k, species, tmodel)
        else:
            tmodel.variables[k] = Variable(value=init_or_zero)
        return

    # STEP 2: branch on initial quantity type
    if species.initial_amount is not None:
        if species.has_only_substance_units:
            if species.has_boundary_condition:
                _handle_amount_boundary_has_substance_units(k, ...)
            else:
                _handle_amount_has_substance_units(k, ...)
        else:
            if species.has_boundary_condition:
                _handle_amount_boundary(k, ...)
            else:
                _handle_amount(k, ...)

    elif species.initial_concentration is not None:
        if variable_is_constant(k, pmodel):
            _handle_constant_variable(k, species, tmodel)
            return
        if species.has_only_substance_units:
            if species.has_boundary_condition:
                _handle_conc_boundary_has_substance_units(k, ...)
            else:
                _handle_conc_has_substance_units(k, ...)
        else:
            if species.has_boundary_condition:
                _handle_conc_boundary(k, ...)
            else:
                _handle_conc(k, ...)

    else:
        # Neither set.
        # [SPEC_SILENT D1] Heuristic fallback: check co-reactants for concentration evidence.
        is_conc = _check_co_reactants_for_concentration_evidence(k, pmodel)
        if is_conc:
            species.initial_concentration = 0.0  # tests: t1513
        else:
            species.initial_amount = 0.0         # tests: t676, t688
        transform_species(k, species, pmodel, tmodel)  # recurse once
```

**`_handle_amount`** — `hasOnlySubstanceUnits=False`, `boundaryCondition=False`

Species identifier appears as **concentration** in kinetic laws; pysbml tracks internally as **amount**.

```
tmodel.variables[k]        = Variable(value=initial_amount)
tmodel.derived["{k}_conc"] = k / compartment

# InitialAssignment fix: IA math gives concentration → multiply by compartment to get amount

# AssignmentRule fix: if k appears in any derived rule, the rule assigns concentration
#   → multiply the rule by compartment; update {k}_conc = k / compartment accordingly

# RateRule fix (if rate rule exists for k):
#   if compartment ALSO has a rate rule (dynamic compartment):
#     d(amount)/dt = d(conc)/dt * C + conc * dC/dt  [chain rule]  [SPEC_SILENT D5]
#     stoichiometry = {k: 1.0}
#   else (constant compartment, rate rule on conc):
#     stoichiometry = {k: compartment_symbol}  (scale by C at ODE assembly time)

# Reaction kinetic laws: substitute k → {k}_conc; remove compartment symbol (set to 1)
#   Deferred stoichiometry: multiply stoich by compartment IF ctx.rxn_had_compartment[rxn]

# EventAssignment fix:
#   triggers/assignment expressions: substitute k → k/compartment
#   assignments targeting k: new_amount = assigned_value * compartment  [SPEC_SILENT D8]
```

**`_handle_amount_has_substance_units`** — `hasOnlySubstanceUnits=True`, `boundaryCondition=False`

Species identifier = amount in all formulas. Also derives `{k}_conc` for callers that need it.

```
tmodel.variables[k]          = Variable(value=initial_amount)
tmodel.derived["{k}_conc"]   = k / compartment
tmodel.substance_units_vars |= {k}  # marks species as amount-identifier in rateOf logic
# Stoichiometry unchanged — kinetic law already uses amounts.
# No InitialAssignment, RateRule, or reaction fixups needed.
```

**`_handle_amount_boundary`** — `hasOnlySubstanceUnits=False`, `boundaryCondition=True`

Reactions do not change this species. Identifier = concentration in formulas.

```
if k not in rate_rules:
    _handle_constant_variable(k, ...)   # removes from reactions, adds to parameters
else:
    tmodel.variables[k] = Variable(value=initial_amount)
tmodel.derived["{k}_conc"] = k / compartment

# InitialAssignment fix: multiply by compartment (conc → amount)
# AssignmentRule fix: multiply rule math by compartment; rederive {k}_conc
# RateRule fix: d(amount)/dt = d(conc)/dt * C + conc * dC/dt  [chain rule, only if C non-constant]
# Reaction fix: substitute k → {k}_conc in kinetic law (no stoichiometry change — boundary)
# EventAssignment fix: same as _handle_amount (k → k/compartment in triggers; assignment → multiply by C)
```

**`_handle_amount_boundary_has_substance_units`** — `hasOnlySubstanceUnits=True`, `boundaryCondition=True`

```
# [SPEC_SILENT D2]
# FIXME in source: despite hasOnlySubstanceUnits=True, this path derives {k}_conc
# and substitutes it into reactions — because test 1123 has a non-boundary species
# whose stoichiometry would otherwise include a compartment that doesn't balance.
if k not in rate_rules:
    _handle_constant_variable(k, ...)
else:
    tmodel.variables[k] = Variable(value=initial_amount)
tmodel.derived["{k}_conc"] = k / compartment

# Reaction fix: substitute k → {k}_conc in kinetic law
# Force ctx.rxn_had_compartment[rxn] = True for all reactions containing k
# (introduces artificial compartment dependency, triggering stoichiometry correction)
```

**`_handle_conc`** — `hasOnlySubstanceUnits=False`, `boundaryCondition=False`, initialConcentration set

```
if compartment.is_constant:
    # Track concentration as the primary variable; derive amount as a convenience
    tmodel.variables[k]            = Variable(value=initial_concentration)
    tmodel.derived["{k}_amount"]   = k * compartment
    # InitialAssignment fix: if compartment has an IA, update species IA to `init * IA_compartment`
    # Stoichiometry: divide by compartment (convert extent/time kinetic law → conc/time)
else:
    # Compartment can change → must track amount to ensure conservation
    k_amount = "{k}_amount"
    tmodel.variables[k_amount]     = Variable(value=initial_concentration)  # raw; IA overrides
    tmodel.derived[k]              = k_amount / compartment
    # InitialAssignment fix: initial_assignments[k_amount] = (ia or init) * compartment
    # Stoichiometry: remap species k → k_amount in all reaction stoichiometries
```

**`_handle_conc_has_substance_units`** — `hasOnlySubstanceUnits=True`, `boundaryCondition=False`, initialConcentration set

Species identifier = amount in formulas, but initial value given as concentration.

```
tmodel.variables[k] = Variable(value=initial_concentration)  # raw
# Set initial_assignments[k] = (ia or init) * compartment  (conc → amount conversion via IA)
# No other fixups needed — kinetic laws already use amount units.
```

**`_handle_conc_boundary`** — `hasOnlySubstanceUnits=False`, `boundaryCondition=True`, initialConcentration set

Concentration is the internal variable; amount is derived. Events must preserve conservation.

```
tmodel.variables["{k}_conc"] = Variable(value=initial_concentration)
tmodel.derived[k]            = {k}_conc * compartment   # k (amount) is derived

# InitialAssignment fix: remap IA from k → {k}_conc
# AssignmentRule fix: rename rules targeting k to target {k}_conc
# RateRule fix: rename fake reaction "d{k}" → "d{k}_conc"; remap stoichiometry k → {k}_conc
# Events fix:
#   - Substitute k → {k}_conc in triggers, delays, priorities, assignment expressions
#   - Assignment targeting k → remap target to {k}_conc
#   - If both species k and compartment assigned simultaneously:
#       {k}_conc_new = formula * C_old / C_new  (conserve amount during instantaneous C change)
#   - If compartment assigned but species not: conserve amount:
#       {k}_conc_new = {k}_conc_old * C_old / C_new  [SPEC_SILENT D12]
```

**`_handle_conc_boundary_has_substance_units`** — `hasOnlySubstanceUnits=True`, `boundaryCondition=True`, initialConcentration set

```
tmodel.variables[k] = Variable(value=initial_concentration)  # raw
# Set initial_assignments[k] = (ia or init) * compartment  (conc → amount conversion)
# AssignmentRule fix: multiply rule math by compartment, BUT only if rule does NOT contain
#   a __rateOf_*__ sentinel (rateOf is already in amount/time units and must not be scaled)
```

### roadrunner

**Internal representation:**

| Aspect                 | pysbml                                                                                       | roadrunner                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Storage                | Explicit `{k}_amount` AND `{k}_conc` as separate model variables; one derived from the other | Amount-canonical: single `FloatingSpeciesAmounts` array; concentration computed at read-time as `amt/vol`       |
| HOSU effect            | Controls which handler is called; determines primary state variable                          | Controls get/set interface only: HOSU=false → divide by compartment on read; HOSU=true → return amount directly |
| Concentration variable | Explicit derived variable `{k}_conc = {k}_amount / C`                                       | Computed on the fly; no separate storage slot                                                                   |
| Boundary species       | Tracked but excluded from ODE; reactions zeroed                                              | Stored in `BoundarySpeciesAmounts`; not in state vector                                                         |

**Decision tree:** pysbml branches on `(HOSU, boundaryCondition, constant, compartment_valid, compartment_constant)` into ~8 `_handle_*` functions. roadrunner uses a uniform path; HOSU only affects load/store conversion at JIT boundaries.

**HOSU=false in kinetic law:** roadrunner loads `amt / compartment` mechanically. Same result as pysbml's D4 heuristic — the compartment factor in the kinetic law and the `amt/compartment` conversion cancel.

**No initialAmount/initialConc:** Injects 0.0 and emits `LOG_WARNING`. Uses HOSU to determine amount vs concentration units; no co-reactant heuristic.

### Divergences

| ID  | Flag            | Library | Description                                                                                                                                                                                                                                                           |
| --- | --------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | `SPEC_SILENT`   | pysbml  | Species with neither `initialAmount` nor `initialConcentration`: spec says value is "unknown or from external source" (§4.6.4). pysbml injects 0.0 using a co-reactant heuristic to guess amount vs concentration. Tests: t676, t688 → amount; t1513 → concentration. |
| D2  | `SPEC_SILENT`   | pysbml  | `_handle_amount_boundary_has_substance_units` (test 1123): source FIXME — despite `hasOnlySubstanceUnits=True`, derives `{k}_conc` and forces `rxn_had_compartment=True`. Spec does not specify this interaction.                                                     |
| D13 | `SPEC_CONFLICT` | pysbml  | `constant=true`, `hasOnlySubstanceUnits=false`, initialized via `initialConcentration` in a non-constant compartment: spec §4.6.4 states the **amount** is held constant. pysbml stores the raw `initialConcentration` value, pinning concentration instead. Bug latent when `C(0)=1` (cases 01117, 01118, 01377). |

---

## 5. Parameter (§4.7)

### Spec

- `value`: optional float. If absent, value is unknown or set by InitialAssignment.
- `constant` (required, default `true`): if `false`, may be changed by rules or events.
- `units`: optional annotation.

### pysbml

`convert_parameters` in `transform/__init__.py`:

```
for each Parameter p:
    if p.is_constant:
        tmodel.parameters[p.id] = Parameter(value=p.value)
    else:
        tmodel.variables[p.id] = Variable(value=p.value)
        # if p.id in rate_rules → becomes fake reaction "d{p.id}"
```

**Event-assigned parameter promotion** (end of `transform()`):

```
# If a parameter in tmodel.parameters is the target of an EventAssignment,
# it is promoted to tmodel.variables.
# (Spec §4.12.5: EventAssignment targets must not be constant=true; pysbml auto-promotes.)
```

### roadrunner

Parameters stored in `GlobalParameters` array; constant parameters optimized at JIT compile time. No divergences noted.

### Divergences

None observed.

---

## 6. InitialAssignment (§4.8)

### Spec

- Overrides the initial value of a symbol (Parameter, Species, Compartment, SpeciesReference stoichiometry).
- Evaluated exactly once at t=0, after all other initial conditions are set.
- All InitialAssignments collectively form an acyclic dependency graph.

### pysbml

`convert_rules_and_initial_assignments` in `transform/__init__.py`:

```
for each InitialAssignment ia targeting symbol s:
    tmodel.initial_assignments[s] = parse_mathml(ia.math)
# Applied as initial conditions, overriding Variable.value.
```

**Species amount correction:** For species handled by `_handle_amount` (identifier = concentration in laws, tracked as amount), an InitialAssignment targeting the species id provides a **concentration** value. pysbml multiplies by compartment size to obtain the initial amount.

### roadrunner

Initial assignments evaluated at model initialization via JIT-compiled code. No divergences noted.

### Divergences

None observed.

---

## 7. AssignmentRule (§4.9.3)

### Spec

- `variable` must be non-constant and must not also be a target of an EventAssignment.
- Math defines the value of `variable` at **all times**.
- A species governed by an assignment rule does not have its amount changed by reactions.

### pysbml

```
for each AssignmentRule ar:
    tmodel.derived[ar.variable] = parse_mathml(ar.math)
    # tmodel.variables / parameters entry for ar.variable is removed by
    # remove_duplicate_entries() called later in the pipeline.
```

### roadrunner

Assignment rules compiled to JIT code evaluated at every ODE step. No divergences noted.

### Divergences

None observed.

---

## 8. RateRule (§4.9.4)

### Spec

- `variable` must be non-constant.
- Math defines `d(variable)/dt`.
- For Species with `hasOnlySubstanceUnits=false`, the species identifier means concentration, so the rate rule gives `d(concentration)/dt`. The rate of change of **amount** must be derived from this.

### pysbml

```
for each RateRule rr targeting variable k:
    # Stored as fake reaction so the ODE framework can add it uniformly
    tmodel.reactions["d{k}"] = Reaction(rate=parse_mathml(rr.math), stoichiometry={k: 1.0})
```

**Dynamic compartment correction** (`_handle_amount`): If species uses `_handle_amount` (identifier = concentration, compartment non-constant) and has a RateRule, the Leibniz product rule is applied:

```
# d(amount)/dt = d(conc)/dt * C + conc * dC/dt  [SPEC_SILENT D5]
```

### roadrunner

Same product rule applied in `EvalRateRuleRatesCodeGen`. Identical result to pysbml for D5.

### Divergences

| ID  | Flag          | Library | Description                                                                                                                                                                                                         |
| --- | ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D5  | `SPEC_SILENT` | pysbml  | Chain rule correction for `hasOnlySubstanceUnits=false` species with rate rule in a dynamic compartment. Spec §4.9.4 does not specify this reconciliation. Both pysbml and roadrunner apply the Leibniz product rule. |

---

## 9. AlgebraicRule (§4.9.2)

### Spec

- Math equals zero: `0 = f(...)`. Defines an implicit algebraic constraint.
- Exactly one "floating variable" — a non-constant symbol undetermined by any other construct — must be identifiable.
- Cannot co-exist with an assignment or rate rule for the same variable.

### pysbml

`convert_algebraic_rules` in `transform/__init__.py`:

```
for each AlgebraicRule ar:
    floating_var = identify_floating_variable(ar.math, pmodel, tmodel)
    solutions = sympy.solve(ar.math_expr, floating_var)
    if solutions:
        tmodel.derived[floating_var] = solutions[0]
    # else: raise or warn (nonlinear case may not yield closed-form solution)
```

### roadrunner

LLVM backend throws `"Unable to support algebraic rules"` on any model containing an algebraicRule. The legacy C generator ignores them. 102 L3v2 test cases (D6) are affected.

### Divergences

| ID   | Flag            | Library    | Description                                                                                                                                                               |
| ---- | --------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D6   | `SPEC_SILENT`   | pysbml     | Spec requires exactly one undetermined variable but does not specify the algorithm. pysbml uses `sympy.solve`, which may fail for nonlinear rules or produce multiple solutions. |
| RR-A | `SPEC_CONFLICT` | roadrunner | LLVM backend throws `"Unable to support algebraic rules"`. Spec §4.8 requires interpreters to support algebraic rules.                                                    |

---

## 10. Constraint (§4.10)

### Spec

- Boolean math expression that should evaluate to `true` at all valid times.
- Optional `message` child: XHTML content for human-readable error.
- Violation is model-defined undefined behavior; interpreters **may** warn or halt.
- Constraints have no mathematical effect on the model.

### pysbml

`convert_constraints` in `transform/__init__.py`:

```
for each Constraint c:
    LOGGER.warning("Constraints are not modelled")
    # No further action.
```

### roadrunner

Behavior not determined from source review.

### Divergences

| ID  | Flag             | Library | Description                                                                                                                                               |
| --- | ---------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D7  | `SPEC_EXTENSION` | pysbml  | Constraints parsed but silently ignored at transform time. Spec says interpreter *may* warn or halt on violation. pysbml issues no warning at solve time. |

---

## 11. Reaction + KineticLaw (§4.11)

### Spec

**Reaction attributes:**

| Attribute    | Required | Default | Meaning                                              |
| ------------ | -------- | ------- | ---------------------------------------------------- |
| `reversible` | yes      | true    | Informational only in L3v2; no mathematical effect   |
| `fast`       | —        | —       | **Removed in L3v2.** Presence is undefined behavior. |

**SpeciesReference:**
- `stoichiometry` (optional, default 1): dimensionless coefficient.
- `constant` (required): if `false`, stoichiometry may change.
- `id` (optional): if set, stoichiometry becomes a dynamic model variable.

**KineticLaw:**
- `math` gives the reaction rate in units of **extent/time** (§4.11.7) — not concentration/time.
- `listOfLocalParameters`: local parameters scoped to this kinetic law; shadow global symbols.
- Local parameters cannot be targets of InitialAssignment, EventAssignment, or Rule.

**Rate of change formula (§4.11.7):**

```
Case 1 — no conversion factor:
    dn_S/dt = Σ_j (stoich_{S,Rj} · v_{Rj})

Case 2 — model-level conversionFactor c_model:
    dn_S/dt = c_model · Σ_j (stoich_{S,Rj} · v_{Rj})

Case 3 — species-level conversionFactor c_S (overrides model-level):
    dn_S/dt = c_S · Σ_j (stoich_{S,Rj} · v_{Rj})
```

### pysbml

**`convert_reactions`:**

```
for each Reaction rxn:
    rate = parse_mathml(rxn.kinetic_law.math)

    # Namespace local parameters: "{rxn.id}_{local_param.id}"
    for lp in rxn.kinetic_law.local_params:
        tmodel.parameters[f"{rxn.id}_{lp.id}"] = Parameter(value=lp.value)

    # Dynamic stoichiometry: SpeciesReference with id → Variable
    for (stoich_val, stoich_id) in species_references:
        if stoich_id is not None:
            tmodel.variables[stoich_id] = Variable(value=stoich_val)

    tmodel.reactions[rxn.id] = Reaction(rate=rate, stoichiometry=signed_stoich_dict)
```

**`rxn_had_compartment` pre-computation** [SPEC_SILENT D4]:

```
# Before transform_species, for each reaction:
ctx.rxn_had_compartment[rxn.id] = (compartment_symbol in kinetic_law_free_symbols)
# If compartment found: remove it from kinetic law; multiply stoichiometry by compartment.
# Detects traditional concentration/time laws and auto-corrects to extent/time.
```

**`apply_conversion_factors`:**

```
for each species k:
    cf_id = species.conversion_factor or model.conversion_factor
    if cf_id:
        for rxn in tmodel.reactions.values():
            if k in rxn.stoichiometry:
                rxn.stoichiometry[k] *= Symbol(cf_id)
```

**`convert_fast_reactions`** [SPEC_EXTENSION D3/D11]:

```
# QSS reduction for reactions with fast=True:
# 1. Identify species exclusively in fast reactions.
# 2. Solve net flux = 0 algebraically (QSS assumption).
# 3. Replace tmodel.variables[sp] with tmodel.derived[sp] = QSS_solution.
# 4. Corrected stoichiometry for non-QSS species via conservation law null space.
# 5. Project initial conditions onto QSS manifold.
# 6. Deferred QSS [D11]: if net_flux = 0 at t=0, keep species as state variable;
#    inject event assignments to snap to QSS when fast reaction is activated.
```

### roadrunner

- **D4 heuristic not needed:** Amount-canonical storage with HOSU-based loading already yields correct ODE. `compartment * k * (amt/compartment) = k * amt` — the factors cancel without pysbml's heuristic.
- **Conservation law reduction (RR-B):** Optional `CONSERVED_MOIETIES` flag enables L0-matrix null-space analysis via `lsLibStructural`. Dependent species rewritten as assignment rules of conserved moiety parameters (`CM_*`). Reduces ODE dimensionality; pysbml has no equivalent.
- **Conversion factors:** `EvalConversionFactorCodeGen` handles model/species-level factors. Same result as pysbml.
- **Dynamic stoichiometry:** `EvalVolatileStoichCodeGen`. Same result.
- **`fast=true`:** Behavior not determined from source review.

### Divergences

| ID   | Flag             | Library    | Description                                                                                                                                                                                                                                                   |
| ---- | ---------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D3   | `SPEC_EXTENSION` | pysbml     | `fast=True` triggers full QSS reduction with conservation-law-based stoichiometry correction. L3v2 spec §4.11 explicitly removed `fast`.                                                                                                                      |
| D4   | `SPEC_SILENT`    | pysbml     | `rxn_had_compartment` heuristic: pysbml auto-detects and auto-corrects traditional concentration/time kinetic laws. Not described in spec; not needed by roadrunner.                                                                                           |
| D11  | `SPEC_EXTENSION` | pysbml     | Deferred QSS: when a fast reaction is inactive at t=0, species kept as state variable; event assignments injected to snap to QSS when the fast reaction activates.                                                                                            |
| RR-B | `SPEC_EXTENSION` | roadrunner | Optional conservation law reduction: L0-matrix null-space analysis rewrites dependent species as `dep = CM - Σ(L0ᵢⱼ · indⱼ)`. Introduces `CM_*` parameters not in original SBML. Numerical behavior equivalent; model structure transformed. |

---

## 12. Event (§4.12)

### Spec

**Structure:**

```
Event
  useValuesFromTriggerTime: boolean  (required, no default)
  Trigger                            (optional)
    initialValue: boolean            (required)
    persistent: boolean              (required)
    math: boolean expression
  Priority                           (optional)
    math: dimensionless number
  Delay                              (optional)
    math: nonneg number, evaluated at trigger time, units = model time
  ListOfEventAssignments             (optional)
    EventAssignment*
      variable: SIdRef
      math: new value
```

**Trigger (§4.12.2, §4.12.7):** Fires on **false → true** transition.
- `initialValue=true` → treated as `true` before t=0; cannot fire at t=0.
- `initialValue=false` → treated as `false` before t=0; can fire at t=0.

**`persistent`:**
- `true` → assignments always execute at delay expiry regardless of trigger state.
- `false` → if trigger goes false before execution, event is cancelled.

**`useValuesFromTriggerTime` (§4.12.1):**
- `true` → EventAssignment math evaluated at trigger time; saved values applied at execution.
- `false` → math evaluated at execution time.

**Priority (§4.12.3):** Higher value executes first. Equal priority → **random order** (spec requires randomness). Priority math re-evaluated after each event execution.

**Delay (§4.12.4):** Math evaluated at trigger time. Execution time = trigger time + delay.

**EventAssignment (§4.12.5):** Variable must not be `constant=true`, a Reaction id, or an AssignmentRule target. Assignment always occurs at execution time. For species: sets the species' quantity (amount if `hasOnlySubstanceUnits=true`, concentration if `false`).

**Cascade (§4.12.7):** After each execution, recheck all triggers. `persistent=false` queued events re-evaluated and cancelled if trigger is false.

### pysbml

`convert_events` in `transform/__init__.py`:

```
for each Event e:
    tmodel.events[e.id] = Event(
        trigger                      = parse_mathml(e.trigger.math),
        delay                        = parse_mathml(e.delay.math) if e.delay else None,
        priority                     = parse_mathml(e.priority.math) if e.priority else None,
        use_values_from_trigger_time = e.use_values_from_trigger_time,
        assignments                  = {ea.variable: parse_mathml(ea.math) for ea in e.event_assignments},
        trigger_initial_value        = e.trigger.initial_value,
        trigger_persistent           = e.trigger.persistent,
    )
```

**`substitute_delays`** [SPEC_SILENT D10]: `SBMLDelay(x, d)` sentinel appears anywhere `delay(x, d)` occurs in MathML — not only in Event Delay elements:

```
For each SBMLDelay(target_expr, delay_amount):
    if target is in derived (AssignmentRule):
        # Substitute time → time - delay_amount in rule expression
    elif target is a static variable (has IA, no dynamics):
        return Piecewise((initial_assignment(time - d), time < d), (x, True))
    elif target is a parameter:
        return target_expr  # constant, delay has no effect
    else:  # true DDE — dynamic variable
        raise NotImplementedError("delay() for dynamic variable not supported (DDE)")
```

**`substitute_rate_of`** [SPEC_SILENT D9]: `__rateOf_X__` sentinel replaced with actual rate expression for symbol X.

**Species event assignment unit conversion** [SPEC_SILENT D8]:

```
# In trigger expressions: substitute k → k / compartment
# EventAssignment targeting k: new_amount = assigned_conc_value * compartment
```

**Test simulator** (`_simulate_events` in `tests/test_import.py`):
- LSODA integration until trigger fires; pending queue with `(execution_time, event_id, saved_values)`.
- `persistent=false`: re-evaluates trigger on each ODE step for queued events.
- Priority: sorts by priority at execution point; random shuffle among ties.
- `useValuesFromTriggerTime`: saves math values at trigger, applies saved values at execution.

### roadrunner

- **EventAssignment to HOSU=false species:** Multiplies by compartment at store time in `ModelDataSymbolResolver::storeSymbolValue`. Identical to D8 behavior.
- **`rateOf` csymbol:** Fully implemented in LLVM codegen. Reconstructs rate from stoichiometry × kinetic laws; quotient-rule correction for HOSU=false species in dynamic compartment. Equivalent to pysbml D9.
- **`delay` csymbol:** Always throws `"Unable to support delay differential equations"`. No fallback. 52 L3v2 test cases (D10) affected.

### Divergences

| ID   | Flag             | Library    | Description                                                                                                                                                                                                                                        |
| ---- | ---------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D8   | `SPEC_SILENT`    | pysbml     | EventAssignment to `hasOnlySubstanceUnits=false` species: spec §4.12.5 does not specify the concentration→amount conversion. pysbml multiplies by compartment. roadrunner does the same at store time (identical result).                           |
| D9   | `SPEC_SILENT`    | pysbml     | `rateOf` csymbol is an L3v2 package feature (not L3v2 Core). Both pysbml and roadrunner implement it; neither rejects it. pysbml via sympy substitution; roadrunner via LLVM codegen with quotient-rule.                                           |
| D10  | `SPEC_EXTENSION` | pysbml     | `delay(x, d)` in kinetic laws / derived rules: spec §4.12.4 defines Delay only within Events. pysbml resolves `SBMLDelay` sentinels in ANY expression via time-shift substitution. True DDEs raise `NotImplementedError`.                           |
| D12  | `SPEC_SILENT`    | pysbml     | `_handle_conc_boundary`: when a compartment EventAssignment fires without assigning the species, pysbml auto-conserves amount via `{k}_conc_new = {k}_conc_old * C_old / C_new`. Spec does not specify this conservation behavior.                  |
| RR-C | `SPEC_CONFLICT`  | roadrunner | `delay` csymbol always throws `"Unable to support delay differential equations"`. Spec §3.4.6 defines delay as a valid MathML operator. 52 L3v2 test cases (D10) would fail.                                                                        |

---

## Appendix A: pysbml Transform Pipeline

Full pipeline in `transform()`:

```
1.  convert_units                    — build unit context
2.  convert_parameters               — constant → param, non-constant → var
3.  convert_compartments             — constant → param, non-constant → var
4.  convert_constraints              — warn + skip [D7]
5.  convert_functions                — FunctionDefinition → sympy Lambda
6.  convert_events                   — trigger/delay/priority/assignments → sympy
7.  convert_rules_and_initial_assignments
                                     — RateRules → fake reactions "d{k}"
                                     — AssignmentRules → derived
                                     — InitialAssignments → initial_assignments
8.  convert_reactions                — kinetic laws → sympy, local params namespaced
9.  pre-compute ctx.rxn_had_compartment  [D4]
10. transform_species                — main species decision tree
11. apply_conversion_factors         — multiply stoichiometry by conversion factor symbols
12. remove_duplicate_entries         — remove params/vars superseded by derived
13. convert_algebraic_rules          — sympy.solve for floating variable [D6]
14. convert_fast_reactions           — QSS reduction [D3]
15. substitute_rate_of               — replace __rateOf_X__ sentinels [D9]
16. substitute_delays                — replace SBMLDelay sentinels [D10]
17. promote event-assigned params to variables
```

---

## Appendix B: pysbml Summary Compliance Table

| Element / Feature                    | Parsed  | Transformed                | L3v2 Tests        | Spec Match       | Divergence     |
| ------------------------------------ | ------- | -------------------------- | ----------------- | ---------------- | -------------- |
| FunctionDefinition                   | ✓       | Lambda → sympy, inlined    | ✓                 | ✓                | —              |
| UnitDefinition                       | ✓       | annotation only            | partial           | ✓                | no enforcement |
| Compartment (constant)               | ✓       | → parameter                | ✓                 | ✓                | —              |
| Compartment (variable)               | ✓       | → variable                 | ✓                 | ✓                | —              |
| Species (amount, HOSU=F, BC=F)       | ✓       | amount + derived conc      | ✓                 | ✓                | —              |
| Species (amount, HOSU=T, BC=F)       | ✓       | amount only                | ✓                 | ✓                | —              |
| Species (amount, HOSU=F, BC=T)       | ✓       | boundary, no rxn ODE       | ✓                 | ✓                | —              |
| Species (amount, HOSU=T, BC=T)       | ✓       | boundary, no rxn ODE       | t1123             | partial          | D2             |
| Species (conc, const compartment)    | ✓       | conc as variable           | ✓                 | ✓                | —              |
| Species (conc, variable compartment) | ✓       | amount + derived conc      | ✓                 | ✓                | —              |
| Species (neither initial)            | ✓       | 0-fallback + heuristic     | t676, t688, t1513 | `SPEC_SILENT`    | D1             |
| Parameter (constant)                 | ✓       | → parameter                | ✓                 | ✓                | —              |
| Parameter (variable)                 | ✓       | → variable                 | ✓                 | ✓                | —              |
| InitialAssignment                    | ✓       | initial value override     | ✓                 | ✓                | —              |
| AssignmentRule                       | ✓       | → derived                  | ✓                 | ✓                | —              |
| RateRule                             | ✓       | → fake reaction "d{k}"     | ✓                 | ✓                | D5             |
| AlgebraicRule                        | ✓       | sympy.solve → derived      | ✓                 | ✓                | D6             |
| Constraint                           | ✓ parse | silently ignored           | partial           | `SPEC_EXTENSION` | D7             |
| Reaction (basic)                     | ✓       | kinetic law → sympy        | ✓                 | ✓                | —              |
| Reaction (fast=True)                 | ✓       | QSS reduction              | ✓                 | `SPEC_EXTENSION` | D3             |
| Reaction (dynamic stoichiometry)     | ✓       | SpeciesRef → variable      | ✓                 | ✓                | —              |
| LocalParameter                       | ✓       | namespaced {rxn}_{par}     | ✓                 | ✓                | —              |
| ConversionFactor (model-level)       | ✓       | stoich multiply            | ✓                 | ✓                | —              |
| ConversionFactor (species-level)     | ✓       | stoich multiply (override) | ✓                 | ✓                | —              |
| rxn_had_compartment heuristic        | —       | auto-detect + correct      | —                 | `SPEC_SILENT`    | D4             |
| Event (trigger, delay, assignments)  | ✓       | → sympy                    | ✓                 | ✓                | —              |
| Trigger initialValue                 | ✓       | stored                     | ✓                 | ✓                | —              |
| Trigger persistent                   | ✓       | stored                     | ✓                 | ✓                | —              |
| Event Priority                       | ✓       | stored                     | ✓                 | ✓                | —              |
| Event Delay (in events)              | ✓       | SBMLDelay sentinel         | ✓                 | ✓                | —              |
| delay() in kinetic laws / rules      | ✓       | SBMLDelay + time-shift     | ✓                 | `SPEC_EXTENSION` | D10            |
| useValuesFromTriggerTime             | ✓       | stored                     | ✓                 | ✓                | —              |
| EventAssignment to species           | ✓       | amount/conc convert        | ✓                 | `SPEC_SILENT`    | D8             |
| Compartment event → conserve amount  | ✓       | auto-adjust {k}_conc       | ✓                 | `SPEC_SILENT`    | D12            |
| rateOf csymbol                       | ✓       | __rateOf_X__ sentinel      | ✓                 | `SPEC_SILENT`    | D9             |
| Deferred QSS events                  | ✓       | inject event assignments   | ✓                 | `SPEC_EXTENSION` | D11            |
| constant species, initialConc, dyn C | ✓ parse | pinned at conc value       | t01117–t01377     | `SPEC_CONFLICT`  | D13            |

---

## Appendix C: Adding a New Library

Add a `### LibraryName` subsection inside each relevant element section:

1. Brief description of how the library handles the element
2. Any divergences from spec or from pysbml behavior

Add a row to the **Feature Matrix** at the top.

Add library-specific divergence IDs (e.g. `TF-A`, `TF-B`) to the element's **Divergences** table with the `Library` column filled in.
