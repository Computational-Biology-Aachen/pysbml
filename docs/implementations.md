# SBML Implementation Comparison

Tracks how each tool transforms SBML L3v2 into an executable ODE system.
Add a new `##` section per library as more tools are compared.

**Spec:** SBML Level 3 Version 2 Core Release 2 (29 March 2019)
**Divergence flags:** See [`divergences.md`](divergences.md) for flag definitions and the full test-case table.

---

## Feature Matrix

| Feature | pysbml | roadrunner |
| --- | --- | --- |
| Internal species storage | Dual: explicit `{k}_amount` + `{k}_conc` variables | Amount-canonical; concentration derived at read time |
| HOSU effect | Selects one of ~8 `_handle_*` dispatch paths | Affects load/store conversion only |
| Algebraic rules | `sympy.solve` → assignment rule (D6) | **Throws** (RR-A) |
| Conservation law reduction | None | Optional L0-matrix moiety analysis (RR-B) |
| Kinetic law compartment factor | D4 auto-strip heuristic | None needed (HOSU loading cancels compartment) |
| EventAssignment to HOSU=false | Multiply by compartment (D8) | Multiply by compartment at store |
| Rate rule + dynamic compartment | Product rule correction (D5) | Same product rule |
| rateOf csymbol | Sentinel + sympy substitution (D9) | Native LLVM IR codegen with quotient-rule |
| delay csymbol | Time-shift approximation (D10) | **Throws** (RR-C) |
| No initialAmount/initialConc | 0.0 + amount-vs-conc heuristic (D1) | 0.0 + LOG\_WARNING |
| Constraints | Silently ignored (D7) | Unknown |
| fast=true reaction | QSS reduction + deferred events (D3/D11) | Unknown |
| Conversion factors | Applied to stoichiometry | EvalConversionFactorCodeGen |
| Dynamic stoichiometry | SpeciesRef → parameter | EvalVolatileStoichCodeGen |

---

## pysbml

**Source:** `src/pysbml/` — parse layer (`parse/`), transform layer (`transform/`)
**Approach:** SBML → sympy ODE/algebraic system. Two-stage pipeline: parse (libsbml → dataclasses) then transform (dataclasses → mxlpy Model).

### Element Decision Trees

## 1. FunctionDefinition (§4.3)

### Spec

- Lambda expression defining a reusable mathematical function.
- Arguments are positional; body is a MathML expression.
- Cannot be recursive (no cycles in function call graph).
- Return type matches body expression type.
- Identifiers of FunctionDefinitions may appear as function calls in MathML `<apply>` elements.

### Transform

`convert_functions` in `transform/__init__.py`:

```
for each FunctionDefinition f with args (a1, ..., an) and body B:
    ctx.functions[f.id] = sympy.Lambda((a1, ..., an), parse_mathml(B))

# During kinetic law / rule parsing, function calls are inlined via substitution.
```

### Tests

Cases 00001–00045 and many others exercise function definitions.

### Divergences

None observed. Implementation matches spec.

---

## 2. UnitDefinition (§4.4)

### Spec

- Named unit composed of base SI units with `kind`, `exponent`, `scale`, `multiplier`.
- Built-in reserved unit names (`substance`, `volume`, `area`, `length`, `time`) override model defaults when defined.
- SBML does not mandate unit enforcement — units are annotation.
- `unitSIdRef` values appear on species, compartments, parameters, model, etc.

### Transform

`convert_units` in `transform/__init__.py`:

- Maps known SBML unit kinds to sympy unit objects.
- Determines `substance_unit` for the model (used for species scaling).
- Does **not** perform runtime unit enforcement.

### Divergences

None relevant to ODE generation.

---

## 3. Compartment (§4.5)

### Spec

- `spatialDimensions`: 0, 1, 2, 3, or unset (packages may allow non-integer).
- `size`: initial value; if absent and no InitialAssignment, value is unknown.
- `constant="true"`: size never changes.
- `constant="false"`: size may change via RateRule, AssignmentRule, or EventAssignment.
- `units`: optional annotation.

### Transform

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

### Divergences

None observed.

---

## 4. Species (§4.6) — CRITICAL

### Spec

Attributes:

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
- The ODE describes rate of change of **amount** (not concentration):
  `dn_S/dt = convFactor_S * Σ_j(stoich_{S,Rj} · v_{Rj})`
- Rate rules on species describe `d(quantity)/dt` where quantity is amount if `hasOnlySubstanceUnits=true`, else concentration.
- `conversionFactor` overrides the model-level `conversionFactor` for this species.

### Decision Tree

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

### Handler Details

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
# AssignmentRule fix: rename rules targeting k to target {k}_conc; rename appearance of k → {k}_conc
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
#   [SPEC_SILENT: rateOf-awareness not in spec]
```

### Divergences

| ID  | Flag          | Description                                                                                                                                                                                                                                                           |
| --- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | `SPEC_SILENT` | Species with neither `initialAmount` nor `initialConcentration`: spec says value is "unknown or from external source" (§4.6.4). pysbml injects 0.0 using a co-reactant heuristic to guess amount vs concentration. Tests: t676, t688 → amount; t1513 → concentration. |
| D2  | `SPEC_SILENT` | `_handle_amount_boundary_has_substance_units` (test 1123): source has FIXME noting unexplained behavioral difference from non-boundary path. Spec does not specify how boundary interacts with `hasOnlySubstanceUnits=True` for amount-initialized species.           |

---

## 5. Parameter (§4.7)

### Spec

- `value`: optional float. If absent, value is unknown or set by InitialAssignment.
- `constant` (required, default `true`): if `false`, may be changed by rules or events.
- `units`: optional annotation.
- A `Parameter` with `constant="false"` and no rule or event assignment leaves its time course undefined.

### Transform

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
# If a parameter that was placed in tmodel.parameters is the target of an EventAssignment,
# it is promoted to tmodel.variables.
# (Spec §4.12.5: EventAssignment targets must not be constant=true; pysbml auto-promotes.)
```

### Divergences

None observed.

---

## 6. InitialAssignment (§4.8)

### Spec

- Overrides the initial value of a symbol (Parameter value, Species `initialAmount`/`initialConcentration`, Compartment `size`, SpeciesReference `stoichiometry`).
- Evaluated exactly once at t=0, after all other initial conditions are set.
- All InitialAssignments collectively form an acyclic dependency graph.
- Can reference any symbol with a defined initial value.

### Transform

`convert_rules_and_initial_assignments` in `transform/__init__.py`:

```
for each InitialAssignment ia targeting symbol s:
    tmodel.initial_assignments[s] = parse_mathml(ia.math)
# Applied as initial conditions, overriding Variable.value.
```

**Species amount correction:** For species handled by `_handle_amount` (identifier = concentration in
laws, tracked as amount), an InitialAssignment targeting the species id provides a **concentration**
value. pysbml multiplies by the compartment size to obtain the initial amount stored in `Variable.value`.

### Divergences

None observed.

---

## 7. AssignmentRule (§4.9.3)

### Spec

- `variable` must be non-constant and must not also be a target of an EventAssignment.
- Math defines the value of `variable` at **all times** (not just t=0).
- Assignment rules collectively with InitialAssignments and KineticLaws must form an acyclic graph.
- A species governed by an assignment rule does not have its amount changed by reactions (the assignment rule overrides).

### Transform

```
for each AssignmentRule ar:
    tmodel.derived[ar.variable] = parse_mathml(ar.math)
    # tmodel.variables / parameters entry for ar.variable is removed by
    # remove_duplicate_entries() called later in the pipeline.
```

### Divergences

None observed.

---

## 8. RateRule (§4.9.4)

### Spec

- `variable` must be non-constant.
- Math defines `d(variable)/dt`.
- Units: {variable quantity} / {time}.
- For Species with `hasOnlySubstanceUnits=false`, the species identifier in formulas means
  concentration, so the rate rule math gives `d(concentration)/dt`. The rate of change of **amount**
  must be derived from this.
- A species with a rate rule is not affected by reactions at the same time (the rate rule replaces
  the reaction contribution for that species; spec does not address combined case).

### Transform

```
for each RateRule rr targeting variable k:
    # Stored as fake reaction so the ODE framework can add it uniformly
    tmodel.reactions["d{k}"] = Reaction(rate=parse_mathml(rr.math), stoichiometry={k: 1.0})
```

**Dynamic compartment + rate rule correction** (`_handle_amount`):

If a species uses `_handle_amount` (identifier = concentration, compartment is non-constant) and
has a RateRule, the Leibniz product rule must be applied:

```
# d(amount)/dt = d(conc)/dt * C + conc * dC/dt
# pysbml computes this transformation. [SPEC_SILENT D5]
```

### Divergences

| ID  | Flag          | Description                                                                                                                                                                                                                                                                                                           |
| --- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D5  | `SPEC_SILENT` | Chain rule correction for dynamic compartment + rate rule on a `hasOnlySubstanceUnits=false` species. Spec §4.9.4 defines rate rule math as `d(variable)/dt` but does not specify how to reconcile this when compartment is time-varying and identifier means concentration. pysbml applies the Leibniz product rule. |

---

## 9. AlgebraicRule (§4.9.2)

### Spec

- Math equals zero: `0 = f(...)`. Defines an implicit algebraic constraint.
- Exactly one "floating variable" — a non-constant symbol whose value is undetermined by any other
  construct — must be identifiable from the rule.
- Cannot co-exist with an assignment or rate rule for the same variable.
- Together with assignment rules and kinetic laws, must form an acyclic assignment graph.

### Transform

`convert_algebraic_rules` in `transform/__init__.py`:

```
for each AlgebraicRule ar:
    # Identify floating variable: the non-constant symbol not determined elsewhere
    floating_var = identify_floating_variable(ar.math, pmodel, tmodel)
    solutions = sympy.solve(ar.math_expr, floating_var)
    if solutions:
        tmodel.derived[floating_var] = solutions[0]
    # else: raise or warn (nonlinear case may not yield closed-form solution)
```

### Divergences

| ID  | Flag          | Description                                                                                                                                                                                                                       |
| --- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D6  | `SPEC_SILENT` | Spec requires exactly one undetermined variable but does not specify the algorithm for finding it or for solving the rule. pysbml uses `sympy.solve`, which may fail for nonlinear algebraic rules or produce multiple solutions. |

---

## 10. Constraint (§4.10)

### Spec

- Boolean math expression that should evaluate to `true` at all valid times.
- Optional `message` child: XHTML content for human-readable error.
- Violation is model-defined undefined behavior; interpreters **may** warn or halt.
- Constraints have no mathematical effect on the model (they are purely declarative).

### Transform

`convert_constraints` in `transform/__init__.py`:

```
for each Constraint c:
    LOGGER.warning("Constraints are not modelled")
    # No further action.
```

### Divergences

| ID  | Flag             | Description                                                                                                                                                                     |
| --- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D7  | `SPEC_EXTENSION` | Constraints are parsed but silently ignored. Spec says an interpreter *may* warn or halt on violation. pysbml issues no warning at solve time and performs no runtime checking. |

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
- `math`: MathML expression giving the reaction rate in units of **extent/time** (§4.11.7).
  - NOT concentration/time — that is the "traditional" rate law unit, which is inappropriate for
    multi-compartment models.
- `listOfLocalParameters`: local parameters scoped to this kinetic law. Identifiers shadow global
  symbols within the kinetic law.
- Local parameters cannot be targets of InitialAssignment, EventAssignment, or Rule.
- Only species that are reactants, products, or modifiers of this reaction may appear by id in the math.

**Rate of change formula (§4.11.7):**

The ODE for species amount n_S (not concentration) is:

```
Case 1 — no conversion factor:
    dn_S/dt = Σ_j (stoich_{S,Rj} · v_{Rj})

Case 2 — model-level conversionFactor c_model (Species has none):
    dn_S/dt = c_model · Σ_j (stoich_{S,Rj} · v_{Rj})

Case 3 — species-level conversionFactor c_S:
    dn_S/dt = c_S · Σ_j (stoich_{S,Rj} · v_{Rj})
    (species-level overrides model-level)
```

where `stoich_{S,Rj}` is the signed net stoichiometry (products positive, reactants negative).

### Transform

**`convert_reactions`:**

```
for each Reaction rxn:
    rate = parse_mathml(rxn.kinetic_law.math)

    # Namespace local parameters: "{rxn.id}_{local_param.id}"
    for lp in rxn.kinetic_law.local_params:
        tmodel.parameters[f"{rxn.id}_{lp.id}"] = Parameter(value=lp.value)
        # substitute lp.id → "{rxn.id}_{lp.id}" in rate expression

    # Dynamic stoichiometry: SpeciesReference with id → non-constant Variable
    for (stoich_val, stoich_id) in species_references:
        if stoich_id is not None:
            tmodel.variables[stoich_id] = Variable(value=stoich_val)
            stoich_symbol = Symbol(stoich_id)
        else:
            stoich_symbol = stoich_val  # float constant

    tmodel.reactions[rxn.id] = Reaction(rate=rate, stoichiometry=signed_stoich_dict)
```

**`rxn_had_compartment` pre-computation** [SPEC_SILENT D4]:

```
# Before transform_species, for each reaction:
ctx.rxn_had_compartment[rxn.id] = (compartment_symbol in kinetic_law_free_symbols)
# Used by _handle_amount to detect whether kinetic law is a traditional conc/time law
# (which includes explicit compartment factor) vs a proper extent/time law.
# If compartment found: remove it from kinetic law; multiply stoichiometry by compartment.
```

**`apply_conversion_factors`:**

```
# After transform_species:
for each species k in model:
    cf_id = species.conversion_factor or model.conversion_factor
    if cf_id:
        cf_sym = Symbol(cf_id)
        for rxn in tmodel.reactions.values():
            if k in rxn.stoichiometry:
                rxn.stoichiometry[k] *= cf_sym
```

**`convert_fast_reactions`** [SPEC_EXTENSION D3]:

```
# QSS reduction for reactions with fast=True:
# 1. Identify species exclusively participating in fast reactions.
# 2. Solve net flux = 0 algebraically for each such species (QSS assumption).
# 3. Replace tmodel.variables[sp] with tmodel.derived[sp] = QSS_solution.
# 4. For non-QSS species also in fast reactions: compute corrected stoichiometry
#    using conservation law null space (c^T * N_fast = 0), via:
#    eff_stoich = (c^T * N_R) / (dT/dS_j)
# 5. Fix initial conditions: project onto QSS manifold using conservation + QSS equations.
# 6. Deferred QSS: if net_flux.subs(params_at_t0) == 0, the fast reaction is inactive
#    at t=0. Keep species as state variable; add event assignments so that when an event
#    activates the fast reaction, the species is instantly set to its QSS value.
#    [SPEC_EXTENSION D11]
```

### Divergences

| ID  | Flag             | Description                                                                                                                                                                                                                                                |
| --- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D3  | `SPEC_EXTENSION` | `fast=True` on Reaction triggers full QSS reduction with conservation-law-based stoichiometry correction. L3v2 spec §4.11 explicitly removed `fast`; its presence is undefined behavior.                                                                   |
| D4  | `SPEC_SILENT`    | `rxn_had_compartment` heuristic: spec defines kinetic law unit as extent/time, but many real models write traditional concentration/time laws including an explicit compartment factor. pysbml auto-detects and auto-corrects this. Not described in spec. |
| D11 | `SPEC_EXTENSION` | Deferred QSS: when a fast reaction is inactive at t=0, species is kept as a state variable and event assignments are injected to snap it to QSS when events activate the fast reaction. Entirely beyond what spec says about `fast`.                       |

---

## 12. Event (§4.12)

### Spec Structure

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
      variable: SIdRef               (Compartment | Species | SpeciesReference | Parameter)
      math: new value
```

### Trigger Semantics (§4.12.2, §4.12.7)

An event fires on a **false → true** transition of the trigger math.

**`initialValue`:**
- `true` → trigger expression is treated as `true` immediately before t=0.
  The event cannot fire at t=0 because there is no transition. It can only fire at t>0
  after the trigger first becomes false, then transitions to true.
- `false` → trigger expression is treated as `false` immediately before t=0.
  The event can fire at t=0 if the trigger expression evaluates to true at that moment.

**`persistent`:**
- `true` → once triggered, assignments always execute when the delay expires,
  regardless of whether the trigger expression is still true.
- `false` → if the trigger expression transitions back to false between trigger time and
  execution time, the event is **cancelled** (removed from the pending queue).
  If the trigger subsequently transitions to true again, that constitutes a new trigger event.

### `useValuesFromTriggerTime` (§4.12.1)

- `true` → EventAssignment math expressions are evaluated at **trigger time**; computed values are
  saved and applied at execution time (after any delay).
- `false` → EventAssignment math expressions are evaluated at **execution time**.
- No default value — must be specified.

### Priority (§4.12.3)

- Higher value executes first among simultaneous events.
- Equal priority → **random order** (spec requires randomness, not merely undefined order).
- No Priority element → undefined order with respect to other no-priority events and with respect to
  events that have a Priority object.

**Priority evaluation:** Priority math is evaluated at execution time, not trigger time.
After executing one event, all remaining simultaneous events have their Priority math re-evaluated
before selecting the next.

### Delay (§4.12.4)

- Delay math evaluated at **trigger time**. Must evaluate to a nonneg number.
- Execution time = trigger time + delay.
- No Delay element → execute immediately (conceptually at trigger time).

### EventAssignment (§4.12.5)

Restrictions:
- `variable` must not be `constant="true"`.
- `variable` must not be a Reaction id.
- A single Event cannot assign the same `variable` twice.
- `variable` cannot also be the target of an AssignmentRule.

Assignment time is always **execution time**. `useValuesFromTriggerTime` controls only when the
math is evaluated, not when the variable is assigned.

For species: EventAssignment sets the species' **quantity** (amount if `hasOnlySubstanceUnits=true`,
concentration if `false`).

### Event Cascade Semantics (§4.12.7)

After each event execution:
- Recheck all trigger expressions (event execution may cause another event to trigger).
- `persistent=false` events in queue: re-evaluate trigger; if false → remove from queue.
- Events can trigger each other forming cascades; cascades may be infinite (simulator should detect).

### Transform

`convert_events` in `transform/__init__.py`:

```
for each Event e:
    trigger  = parse_mathml(e.trigger.math)
    delay    = parse_mathml(e.delay.math) if e.delay else None
    priority = parse_mathml(e.priority.math) if e.priority else None
    assignments = {ea.variable: parse_mathml(ea.math) for ea in e.event_assignments}

    tmodel.events[e.id] = Event(
        trigger                    = trigger,
        delay                      = delay,      # SBMLDelay sentinel before substitute_delays()
        priority                   = priority,
        use_values_from_trigger_time = e.use_values_from_trigger_time,
        assignments                = assignments,
        trigger_initial_value      = e.trigger.initial_value,
        trigger_persistent         = e.trigger.persistent,
    )
```

**`substitute_delays`** [SPEC_SILENT D10]:

The `SBMLDelay(x, d)` sentinel appears not only in event delays but anywhere `delay(x, d)` appears
in MathML (kinetic laws, derived rules, initial assignments). `substitute_delays` resolves these:

```
For each SBMLDelay(target_expr, delay_amount):
    if target is in derived (AssignmentRule):
        # Substitute time → time - delay_amount in the rule expression (exact, no DDE)
    elif target is a static variable (has InitialAssignment, no dynamics):
        # Piecewise: history(t-d) for t < d, else current value
        return Piecewise((initial_assignment(time - d), time < d), (x, True))
    elif target is a parameter:
        return target_expr  (constant, delay has no effect)
    else:  # true DDE — dynamic variable
        raise NotImplementedError("delay() for dynamic variable not supported (DDE)")
```

**`substitute_rate_of`:**

The `rateOf` csymbol (an L3v2 package mechanism) produces a sentinel `__rateOf_X__` during parsing.
`substitute_rate_of` replaces these with the actual rate expression for symbol X. [SPEC_SILENT D9]

**Event-assigned parameter promotion:**

```
# At end of transform():
for each Event e, EventAssignment ea:
    if ea.variable in tmodel.parameters:
        # Parameter is target of event assignment → must be mutable
        val = tmodel.parameters.pop(ea.variable)
        tmodel.variables[ea.variable] = Variable(value=val.value)
```

**Species event assignment unit conversion** [SPEC_SILENT D8]:

For species handled by `_handle_amount` (identifier = concentration in formulas, tracked as amount):

```
# In trigger expressions: substitute k → k / compartment
# For EventAssignment targeting k:
#   assigned value is in concentration units → convert to amount: multiply by compartment
#   new_amount = assigned_conc_value * compartment
```

**Test simulator** (`_simulate_events` in `tests/test_import.py`):

- Integrates ODE with LSODA until trigger fires.
- Maintains a pending queue with (execution_time, event_id, saved_values_if_trigger_time).
- `persistent=false`: reevaluates trigger on each ODE step for queued events; cancels if false.
- Priority: at each execution point, sorts pending events by priority; random shuffle among ties.
- `useValuesFromTriggerTime`: saves math values at trigger, applies the saved values at execution.

### Divergences

| ID  | Flag             | Description                                                                                                                                                                                                                                                                                                           |
| --- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D8  | `SPEC_SILENT`    | EventAssignment to `hasOnlySubstanceUnits=false` amount-tracked species: spec §4.12.5 says math must match species quantity unit but does not specify the conversion. pysbml multiplies by compartment to convert concentration → amount.                                                                             |
| D9  | `SPEC_SILENT`    | `rateOf` csymbol supported via `__rateOf_X__` sentinel + `substitute_rate_of()`. `rateOf` is an L3v2 package feature, not L3v2 Core. Silently accepted rather than rejected.                                                                                                                                          |
| D10 | `SPEC_EXTENSION` | `delay(x, d)` in kinetic laws / derived rules: spec §4.12.4 defines Delay only within Events. pysbml resolves `SBMLDelay` sentinels in ANY expression (kinetic laws, derived, IAs) via time-shift substitution. True DDEs (dynamic variables) raise NotImplementedError.                                              |
| D12 | `SPEC_SILENT`    | `_handle_conc_boundary` event conservation: when a compartment EventAssignment fires without assigning the species, pysbml auto-conserves amount by adjusting `{k}_conc_new = {k}_conc_old * C_old / C_new`. Spec does not specify this conservation behavior for simultaneous compartment/species event assignments. |

---


### Transform Pipeline Order

## Transform Pipeline Order

The full pipeline in `transform()` (relevant to understanding interaction between steps):

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
10. transform_species                — main species decision tree (§4 above)
11. apply_conversion_factors         — multiply stoichiometry by conversion factor symbols
12. remove_duplicate_entries         — remove params/vars superseded by derived
13. convert_algebraic_rules          — sympy.solve for floating variable [D6]
14. convert_fast_reactions           — QSS reduction [D3]
15. substitute_rate_of               — replace __rateOf_X__ sentinels [D9]
16. substitute_delays                — replace SBMLDelay sentinels
17. promote event-assigned params to variables
```

---


### Summary Compliance Table

## Summary Compliance Table

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


---

## roadrunner

**Source:** `ref/roadrunner/source/` (LLVM backend)
**Approach:** JIT-compiles SBML to native machine code via LLVM IR. Species stored as amounts; concentration derived at read/write boundaries.

## RoadRunner Comparison

**Source:** `ref/roadrunner/source/` (LLVM backend, commit in `ref/roadrunner/`).
**Architecture:** roadrunner JIT-compiles SBML to native code via LLVM IR. pysbml converts SBML to a sympy-based ODE/algebraic system.

### Internal Species Representation

| Aspect | pysbml | roadrunner |
| --- | --- | --- |
| Storage | Explicit `{k}_amount` AND `{k}_conc` as separate model variables; one derived from the other | Amount-canonical: single `FloatingSpeciesAmounts` array; concentration computed at read-time as `amt/vol` |
| HOSU effect | Controls which handler is called; determines primary state variable | Controls get/set interface only: HOSU=false → divide by compartment on read; HOSU=true → return amount directly |
| Concentration variable | Explicit derived variable `{k}_conc = {k}_amount / C` | Computed on the fly; no separate storage slot |
| Boundary species | Tracked but excluded from ODE; reactions zeroed | Stored in `BoundarySpeciesAmounts`; not in state vector |

### Decision Tree Comparison: Species

**pysbml** branches on `(HOSU, boundaryCondition, constant, compartment_valid, compartment_constant)` → selects one of ~8 `_handle_*` functions, each creating its own combination of state variables and derived variables.

**roadrunner**: uniform path regardless of HOSU/BC. All species stored as amounts. The HOSU flag only affects the load/store conversion at JIT-compiled boundaries.

### Feature-by-Feature Comparison

| Feature | pysbml behavior | roadrunner behavior | Same result? |
| --- | --- | --- | --- |
| `HOSU=false` species in kinetic law | tracks as concentration; D4 heuristic strips compartment factor if present | loads `amt / compartment` (concentration) mechanically | Yes — both yield concentration; D4 heuristic compensates for same pattern |
| `HOSU=true` species in kinetic law | tracks as amount directly | loads raw amount | Yes |
| EventAssignment to `HOSU=false` species | D8: multiply assigned value by compartment to get amount | multiplies by compartment at store time (`ModelDataSymbolResolver::storeSymbolValue`) | Yes — identical conversion |
| Rate rule on `HOSU=false` species, dynamic compartment | D5: product rule `dA/dt = V·dc/dt + c·dV/dt` | same product rule (`EvalRateRuleRatesCodeGen`) | Yes — identical chain rule |
| Algebraic rules | D6: `sympy.solve` identifies floating variable; supports nonlinear only if sympy can solve | **THROWS** `"Unable to support algebraic rules"` (LLVM backend); legacy C generator ignores them | No — pysbml handles algebraic rules; roadrunner rejects them |
| Conservation law reduction | None; all floating species are ODE state variables | Optional (`CONSERVED_MOIETIES` flag): L0-matrix null-space analysis via `lsLibStructural`; dependent species rewritten as assignment rules of conserved moiety parameters | No — roadrunner can reduce ODE dimensionality; pysbml cannot |
| `rateOf` csymbol | D9: sentinel `__rateOf_X__` + sympy substitution via `substitute_rate_of()` | Fully implemented in LLVM codegen: reconstructs rate from stoichiometry × kinetic laws; quotient-rule correction for HOSU=false species in dynamic compartment | Equivalent (both implement the correct rateOf semantics) |
| `delay` csymbol | D10: time-shift substitution (`t → t−d`) for assignment rules; `Piecewise` for static vars; `NotImplementedError` for true DDEs | **THROWS** `"Unable to support delay differential equations"` — no fallback at all | No — pysbml partially supports delay; roadrunner always rejects |
| No `initialAmount`/`initialConcentration` | D1: injects 0.0 using co-reactant heuristic (amount vs concentration) | Injects 0.0 and emits `LOG_WARNING`; no amount-vs-concentration heuristic (uses HOSU to determine units) | Similar — both default to 0; roadrunner warns, pysbml guesses |
| `fast=true` reaction | D3/D11: full QSS reduction + deferred-QSS event injection | Unknown from source review (likely ignored or rejected in L3v2 context) | Unknown |
| Constraints | D7: silently ignored | Unknown | Unknown |
| Conversion factors | Applied to stoichiometry | `EvalConversionFactorCodeGen` handles model/species-level conversion factors | Same |
| Dynamic stoichiometry (`SpeciesReference` with `id`) | Tracked as parameter; substituted into ODE | `EvalVolatileStoichCodeGen` generates code for runtime stoichiometry | Same |

### Roadrunner-Specific Divergences

| ID | Flag | Element | Description |
| --- | --- | --- | --- |
| RR-A | `SPEC_CONFLICT` | AlgebraicRule | roadrunner LLVM backend throws `"Unable to support algebraic rules"` on any model containing an algebraicRule. The SBML L3v2 spec §4.8 requires interpreters to support algebraic rules. 102 L3v2 test cases (D6) would fail. |
| RR-B | `SPEC_EXTENSION` | Species/Reaction | Optional conservation law reduction via `CONSERVED_MOIETIES` flag: L0-matrix null-space analysis rewrites dependent species as `dep = CM - Σ(L0ᵢⱼ · indⱼ)`. Introduces new global parameters (`CM_*`) not present in original SBML. Numerical behavior is equivalent but model structure is transformed. |
| RR-C | `SPEC_CONFLICT` | Event / MathML | `delay` csymbol always throws `"Unable to support delay differential equations"`. Spec §3.4.6 defines delay as a valid MathML operator. 52 L3v2 test cases (D10) would fail. |

### Architectural Summary

pysbml and roadrunner reach the same ODE for most compliant L3v2 models. Key behavioral gaps:

1. **Algebraic rules**: roadrunner rejects (throw); pysbml solves via sympy.
2. **Delay**: roadrunner rejects (throw); pysbml approximates via time-shift.
3. **D4 heuristic**: pysbml auto-corrects traditional concentration/time kinetic laws; roadrunner does not need the heuristic because its amount-canonical representation with HOSU-based loading already yields the correct ODE (the compartment factor and the concentration conversion cancel).
4. **Species variables**: pysbml exposes both `{k}_amount` and `{k}_conc` as named model variables; roadrunner only exposes the amount internally (concentration is derived at read time and not a named variable).
5. **Conservation laws**: roadrunner can optionally reduce ODE dimension via moiety analysis; pysbml cannot.

---



---

## Template for future libraries

When adding a new library, add a `## LibraryName` section with:

1. **Source / version** being compared
2. **Architecture** (one sentence: how it converts SBML to executable form)
3. **Feature-by-feature table** (same rows as the matrix above)
4. **Library-specific divergences** table:

| ID | Flag | Element | Description |
| --- | --- | --- | --- |
| XX-A | `SPEC_CONFLICT/SPEC_SILENT/SPEC_EXTENSION` | Element | What it does vs what spec says |

5. **Architectural summary** (bullet list of key differences from pysbml/spec)
