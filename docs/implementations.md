# SBML Implementation Comparison

Tracks how each tool transforms SBML L3v2 into an executable ODE system.
Organized by SBML element; each section has a `### LibraryName` subsection per tool.
Add a new `### LibraryName` block inside each element when comparing more tools.

**Spec:** SBML Level 3 Version 2 Core Release 2 (29 March 2019)
**Divergence flags:** See [`divergences.md`](divergences.md) for flag definitions and the full test-case table.

---

## Feature Matrix

| Feature                         | pysbml                                             | roadrunner                                           | copasi                                                        | pysces                                                                     | SBMLToolkit.jl                                                    |
| ------------------------------- | -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Internal species storage        | Dual: explicit `{k}_amount` + `{k}_conc` variables | Amount-canonical; concentration derived at read time | Concentration-canonical; amount derived as `conc × vol`       | Mode-switched: conc unless ANY HOSU=true → amount-canonical for all (PS-F) | Amount-canonical; initialConc converted at import                 |
| HOSU effect                     | Selects one of ~8 `_handle_*` dispatch paths       | Affects load/store conversion only                   | HOSU=true → multiply by vol in kinetic laws, divide elsewhere | Triggers model-wide amount mode if any species has HOSU=true               | HOSU=false → divide by vol in kinetic law; HOSU=true → raw amount |
| Algebraic rules                 | `sympy.solve` → assignment rule (D6)               | **Throws** (RR-A)                                    | **Silently ignores** (CP-A)                                   | Silently ignores with error log (PS-C)                                     | Algebraic constraint `0 ~ rhs` (MTK DAE)                          |
| Conservation law reduction      | None                                               | Optional L0-matrix moiety analysis (RR-B)            | None                                                          | None                                                                       | None                                                              |
| Kinetic law compartment factor  | D4 auto-strip heuristic                            | None needed (HOSU loading cancels compartment)       | Divides ALL kinetic laws by vol at import (CP-F)              | None (assumes rates already correct)                                       | HOSU=false species divided by vol (`extensive_kinetic_math`)      |
| EventAssignment to HOSU=false   | Multiply by compartment (D8)                       | Multiply by compartment at store                     | Direct assignment (concentration is native)                   | No conversion applied                                                      | Multiply by compartment                                           |
| EventAssignment to HOSU=true    | Direct (amount is native)                          | Direct (amount is native)                            | Divide by compartment (amount → concentration)                | No conversion applied                                                      | Direct (amount is native)                                         |
| Rate rule + dynamic compartment | Product rule correction (D5)                       | Same product rule                                    | Not documented                                                | No product rule                                                            | Product rule applied (D(S) ~ C·f + S/C·D(C))                      |
| rateOf csymbol                  | Sentinel + sympy substitution (D9)                 | Native LLVM IR codegen with quotient-rule            | Auxiliary parameter workaround (CP-C)                         | Not implemented                                                            | Native: `rateOf` → `D()` (MTK derivative)                         |
| delay csymbol                   | Time-shift approximation (D10)                     | **Throws** (RR-C)                                    | Auxiliary parameter workaround (CP-B)                         | Stripped from kinetic law, not replaced (PS-E)                             | **Throws** (JL-B)                                                 |
| No initialAmount/initialConc    | 0.0 + amount-vs-conc heuristic (D1)                | 0.0 + `LOG_WARNING`                                  | Unknown                                                       | 0.0 (libsbml default, silent)                                              | 0 (SBML.jl default)                                               |
| Constraints                     | Silently ignored (D7)                              | Unknown                                              | Warns and ignores (CP-D)                                      | Fatal error; model not loaded (PS-B)                                       | **Throws** (JL-A)                                                 |
| InitialAssignment               | Supported                                          | Supported                                            | Supported                                                     | Fatal error; model not loaded (PS-A)                                       | Supported                                                         |
| fast=true reaction              | QSS reduction + deferred events (D3/D11)           | Unknown                                              | Converted to normal reaction, no QSS (CP-E)                   | Ignored with warning                                                       | Silently treated as normal reaction                               |
| Conversion factors              | Applied to stoichiometry                           | EvalConversionFactorCodeGen                          | Unknown                                                       | Not implemented                                                            | Not implemented                                                   |
| Dynamic stoichiometry           | SpeciesRef → parameter                             | EvalVolatileStoichCodeGen                            | Stoichiometric expression map                                 | stoichiometryMath rejected                                                 | Local params promoted to global                                   |

### Library Architectures

**pysbml** — `src/pysbml/` — parse layer (`parse/`), transform layer (`transform/`)
SBML → sympy ODE/algebraic system via two-stage pipeline (parse: libsbml → dataclasses; transform: dataclasses → mxlpy Model).

**roadrunner** — `ref/roadrunner/source/` (LLVM backend)
JIT-compiles SBML to native machine code via LLVM IR. Amount-canonical species storage; concentration derived at read/write boundaries.

**copasi** — `ref/copasi/copasi/sbml/SBMLImporter.cpp`
SBML → COPASI internal model (CModel). Concentration-canonical species storage throughout. Kinetic laws divided by compartment volume at import; all internal ODEs track concentration, not amount.

**pysces** — `ref/pysces/pysces/core2/PyscesCore2Interfaces.py` (`SbmlToCore` class, line 1222)
SBML → Core2 object representation (dictionaries `__sDict__`, `__nDict__`, `__rules__`, `__eDict__`), then compiled Python rate equations and ODE definitions. Operates in concentration space unless any species has `hasOnlySubstanceUnits=true`, which switches the entire model to amount-canonical mode.

**SBMLToolkit.jl** — `ref/SBMLToolkit.jl/src/` (Julia)
SBML → `Catalyst.ReactionSystem` / `ModelingToolkit.ODESystem`. Amount-canonical; `initialConcentration` converted to amount at import via `SBML.initial_amounts(model, convert_concentrations=true)`. Events become `ContinuousVectorCallback` pairs.

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

### copasi

`createCFunctionFromFunctionDefinition()` and `createCFunctionFromFunctionTree()` (line 1345/1424). Functions mapped to COPASI CFunction objects. COPASI detects functions annotated with special URIs (RATE, RNORMAL, RUNIFORM, RGAMMA, RPOISSON) and replaces them with built-in stochastic distribution functions. Time-dependent functions are detected and time is added as an extra parameter (line 1501).

### pysces

`importFunctionDefinitions()` (line 1687–1726). Functions parsed via `sbmlFormulaToInfix()` (line 1691), stored in `__functions__` with arguments and compiled formulas, then added to NewCore via `addFunctions()` and compiled to Python via InfixParser. Time csymbols supported. No divergences noted.

### SBMLToolkit.jl

FunctionDefinitions are inlined by preprocessing (`convert_promotelocals_expandfuns`, line 126 in `systems.jl`) before symbolic parsing begins. If a lambda expression somehow reaches the symbolic layer, `utils.jl` line 22 throws `ErrorException("Symbolics.jl does not support lambda functions")` as a safety net. In practice, all FunctionDefinitions are inlined and no error is raised.

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

### copasi

Units extracted and stored on CModelValue and CCompartment objects via `setUnitExpression()`. Non-integer spatial dimensions trigger warnings but are rounded down (lines 1576–1580). No runtime unit enforcement.

### pysces

Units stored in `__uDict__` with multiplier, exponent, scale, and kind (lines 1407–1430). Default units (mole, litre, second) assigned to substance, volume, time. Only single-unit definitions stored (multi-unit definitions partially ignored, line 1409). No runtime unit enforcement.

### SBMLToolkit.jl

Units carried in SBML.jl metadata but not converted to symbolic constraints or validation checks. Annotation-only; no runtime enforcement.

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

### copasi

`createCCompartmentFromCompartment()` (line 1545). Spatial dimensions stored on CCompartment; defaults to 3D if missing. For SBML L1/L2, compartments with `spatialDimensions=0` imply `hasOnlySubstanceUnits=true` for contained species (lines 1698–1703). No divergences from L3v2 spec noted.

### pysces

Compartments stored in `__compartments__` with size, dimensions, and optional outside reference (lines 1538–1583). **COMP_FUDGE_FACTOR heuristic** (lines 1557–1578): compartments smaller than 1e-6 are rescaled to prevent numerical issues; amounts are rescaled accordingly. No dynamic compartment support — all compartments treated as constant in ODE generation.

### SBMLToolkit.jl

Constant compartments → `create_param(k)` (line 86 in `utils.jl`). Dynamic compartments → `create_var(k, IV; isbcspecies=true)` (line 87). Zero-dimensional compartments always treated as parameters (line 39–43 in `rules.jl`). Product rule applied automatically when a species RateRule references a dynamic compartment.

### Divergences

| ID   | Flag          | Library | Description                                                                                                                               |
| ---- | ------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| PS-G | `SPEC_SILENT` | pysces  | COMP_FUDGE_FACTOR rescales compartments smaller than 1e-6. Non-standard heuristic not described in spec; affects initial species amounts. |

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

```
tmodel.variables[k]          = Variable(value=initial_amount)
tmodel.derived["{k}_conc"]   = k / compartment
tmodel.substance_units_vars |= {k}
```

**`_handle_amount_boundary`** — `hasOnlySubstanceUnits=False`, `boundaryCondition=True`

```
if k not in rate_rules:
    _handle_constant_variable(k, ...)
else:
    tmodel.variables[k] = Variable(value=initial_amount)
tmodel.derived["{k}_conc"] = k / compartment
```

**`_handle_conc`** — `hasOnlySubstanceUnits=False`, `boundaryCondition=False`, initialConcentration set

```
if compartment.is_constant:
    tmodel.variables[k]            = Variable(value=initial_concentration)
    tmodel.derived["{k}_amount"]   = k * compartment
else:
    tmodel.variables["{k}_amount"] = Variable(value=initial_concentration)  # IA overrides
    tmodel.derived[k]              = "{k}_amount" / compartment
```

**`_handle_conc_boundary`** — `hasOnlySubstanceUnits=False`, `boundaryCondition=True`, initialConcentration set

```
tmodel.variables["{k}_conc"] = Variable(value=initial_concentration)
tmodel.derived[k]            = {k}_conc * compartment
# Events: if compartment assigned but species not → conserve amount:
#   {k}_conc_new = {k}_conc_old * C_old / C_new  [SPEC_SILENT D12]
```

### roadrunner

**Internal representation:**

| Aspect                 | pysbml                                                                                       | roadrunner                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Storage                | Explicit `{k}_amount` AND `{k}_conc` as separate model variables; one derived from the other | Amount-canonical: single `FloatingSpeciesAmounts` array; concentration computed at read-time as `amt/vol`       |
| HOSU effect            | Controls which handler is called; determines primary state variable                          | Controls get/set interface only: HOSU=false → divide by compartment on read; HOSU=true → return amount directly |
| Concentration variable | Explicit derived variable `{k}_conc = {k}_amount / C`                                        | Computed on the fly; no separate storage slot                                                                   |
| Boundary species       | Tracked but excluded from ODE; reactions zeroed                                              | Stored in `BoundarySpeciesAmounts`; not in state vector                                                         |

**Decision tree:** pysbml branches into ~8 `_handle_*` functions. roadrunner uses a uniform path; HOSU only affects load/store conversion at JIT boundaries.

### copasi

`createCMetabFromSpecies()` (line 1641). Concentration-canonical. HOSU=true → tracked in `mSubstanceOnlySpecies`; expressions compensated by volume. EventAssignment: HOSU=true → divide by vol; HOSU=false → direct.

### pysces

`SbmlToCore` class (line 1589–1636). **Model-wide HOSU mode** (PS-F): scans all species before processing; if ANY has HOSU=true, sets `SPECIES_IN_AMOUNTS=True` and switches all species to amount-canonical. In amount mode, initial amounts divided by COMP_FUDGE_FACTOR if compartment was rescaled. `boundaryCondition=true` OR `constant=true` → `fixed=True` (line 1617–1624); no ODE term generated. EventAssignments stored as raw variable→formula pairs with no amount/concentration conversion.

### SBMLToolkit.jl

`SBML.initial_amounts(model, convert_concentrations=true)` (line 196 in `systems.jl`). Amount-canonical; initialConcentration converted to `initialAmount = C₀ × conc` at import. HOSU=false species divided by compartment in kinetic laws (`extensive_kinetic_math`, line 59 in `rules.jl`). HOSU=true species left unchanged. Boundary species added to reaction product lists as pseudo-products (line 118–126 in `reactions.jl`) to prevent net consumption. EventAssignment to HOSU=false species multiplied by compartment (line 22 in `events.jl`).

### Divergences

| ID   | Flag            | Library | Description                                                                                                                                                                                                                                                                                                        |
| ---- | --------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D1   | `SPEC_SILENT`   | pysbml  | Species with neither `initialAmount` nor `initialConcentration`: spec says value is "unknown or from external source" (§4.6.4). pysbml injects 0.0 using a co-reactant heuristic to guess amount vs concentration. Tests: t676, t688 → amount; t1513 → concentration.                                              |
| D2   | `SPEC_SILENT`   | pysbml  | `_handle_amount_boundary_has_substance_units` (test 1123): source FIXME — despite `hasOnlySubstanceUnits=True`, derives `{k}_conc` and forces `rxn_had_compartment=True`. Spec does not specify this interaction.                                                                                                  |
| D13  | `SPEC_CONFLICT` | pysbml  | `constant=true`, `hasOnlySubstanceUnits=false`, initialized via `initialConcentration` in a non-constant compartment: spec §4.6.4 states the **amount** is held constant. pysbml stores the raw `initialConcentration` value, pinning concentration instead. Bug latent when `C(0)=1` (cases 01117, 01118, 01377). |
| PS-F | `SPEC_SILENT`   | pysces  | Model-wide HOSU mode: if ANY species has `hasOnlySubstanceUnits=true`, ALL species switch to amount-canonical. Spec defines HOSU per-species; pysces treats it as a global model flag.                                                                                                                             |

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

**Event-assigned parameter promotion** (end of `transform()`): parameters that are targets of EventAssignments are promoted to variables.

### roadrunner

Parameters stored in `GlobalParameters` array; constant parameters optimized at JIT compile time. No divergences noted.

### copasi

`createCModelValueFromParameter()` (line 3320). Creates a COPASI CModelValue per parameter; units stored via `setUnitExpression()`. No divergences from spec noted.

### pysces

Parameters extracted via `getListOfParameters()`, stored in `init_par` dict (line 1670). NaN values reset to 0.0 with warning (lines 1672–1676). Parameters with units logged as errors (lines 1678–1684) but otherwise stored normally. Parameters targeted by AssignmentRules become `SpeciesAssignmentRule` objects.

### SBMLToolkit.jl

`constant=true` → `create_param(k)`. `constant=false` AND (`seemsdefined` OR is event assignment target) → `create_var(k, IV; isbcspecies=true)` (line 153–160 in `utils.jl`). Event-assigned parameters get zero-rate ODEs: `D(var) ~ 0` (line 100 in `systems.jl`).

### Divergences

None observed.

---

## 6. InitialAssignment (§4.8)

### Spec

- Overrides the initial value of a symbol (Parameter, Species, Compartment, SpeciesReference stoichiometry).
- Evaluated exactly once at t=0, after all other initial conditions are set.
- All InitialAssignments collectively form an acyclic dependency graph.

### pysbml

`convert_rules_and_initial_assignments` in `transform/__init__.py`. For species handled by `_handle_amount`, InitialAssignment provides a concentration value; pysbml multiplies by compartment size to obtain the initial amount.

### roadrunner

Initial assignments evaluated at model initialization via JIT-compiled code. No divergences noted.

### copasi

`importInitialAssignments()` (line 6324). HOSU=true species or zero-dimensional compartments: expression divided by vol (lines 6411–6427). Species references (L3) stored in `mStoichiometricExpressionMap`.

### pysces

**Not supported.** `checkSbmlSupport()` detects InitialAssignments and appends a fatal error: `"PySCeS does not support InitialAssignments"`. Import continues but is flagged. No workaround applied.

### SBMLToolkit.jl

Evaluated at t=0; stored in `initial_assignments` dict (line 232–235 in `systems.jl`). Overrides `u0map` and `parammap`. May reference other species/parameters; handled via `initial_conditions` dict to permit non-parameter references (lines 133–138).

### Divergences

| ID   | Flag            | Library | Description                                                                                                 |
| ---- | --------------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| PS-A | `SPEC_CONFLICT` | pysces  | InitialAssignments cause a fatal error; model import flagged. Spec §4.8 requires InitialAssignment support. |

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
```

### roadrunner

Assignment rules compiled to JIT code evaluated at every ODE step. No divergences noted.

### copasi

`importSBMLRule()` → `importRuleForModelEntity()` (line 5577). HOSU=true species: expression divided by vol (lines 5695–5710). Species reference rules converted to stoichiometric expression storage.

### pysces

Detected via `rule.isAssignment()` (line 1984), stored in `__rules__` with type='assignment'. At runtime, implemented as `SpeciesAssignmentRule` objects that compile and execute Python code via `exec()`. Evaluated in `_Function_forced`. No symbolic Jacobian differentiation.

### SBMLToolkit.jl

Converted to observed-variable equation `var ~ rhs` (line 11 in `rules.jl`). Added to ODESystem as an algebraic constraint. RHS substituted into `defs` dict (line 76 in `systems.jl`). Volume correction applied for HOSU=false species (line 80–82 in `rules.jl`).

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
tmodel.reactions["d{k}"] = Reaction(rate=parse_mathml(rr.math), stoichiometry={k: 1.0})
# Dynamic compartment: d(amount)/dt = d(conc)/dt * C + conc * dC/dt  [SPEC_SILENT D5]
```

### roadrunner

Same product rule applied in `EvalRateRuleRatesCodeGen`. Identical result to pysbml for D5.

### copasi

Rate rules detected via `SBML_RATE_RULE` (line 5334). Same HOSU handling as assignment rules. Rate rules on species references skipped (line 5377).

### pysces

Detected via `rule.isRate()` (line 1986), stored with type='rate'. Implemented as `RateRule` objects (line 535 in `PyscesCore2.py`) that compute rates via `exec()`. **No product rule for dynamic compartments**: raw rate applied without volume compensation.

### SBMLToolkit.jl

Converted to `D(var) ~ rhs` (line 14 in `rules.jl`). Variable created with `isbcspecies=true` (line 78) to prevent conservation law reduction. **Product rule applied for HOSU=false in dynamic compartment** (lines 88–92):

```julia
D(S) ~ C * f + S/C * D(C)   # where f = rhs, C = compartment
```

### Divergences

| ID  | Flag          | Library | Description                                                                                                                                                                                                                                             |
| --- | ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D5  | `SPEC_SILENT` | pysbml  | Chain rule correction for `hasOnlySubstanceUnits=false` species with rate rule in a dynamic compartment. Spec §4.9.4 does not specify this reconciliation. Both pysbml, roadrunner, and SBMLToolkit.jl apply the Leibniz product rule; pysces does not. |

---

## 9. AlgebraicRule (§4.9.2)

### Spec

- Math equals zero: `0 = f(...)`. Defines an implicit algebraic constraint.
- Exactly one "floating variable" — a non-constant symbol undetermined by any other construct — must be identifiable.
- Cannot co-exist with an assignment or rate rule for the same variable.

### pysbml

```
floating_var = identify_floating_variable(ar.math, pmodel, tmodel)
solutions = sympy.solve(ar.math_expr, floating_var)
if solutions:
    tmodel.derived[floating_var] = solutions[0]
```

### roadrunner

LLVM backend throws `"Unable to support algebraic rules"`. 102 L3v2 test cases (D6) affected.

### copasi

Sets `mUnsupportedRuleFound=true` (line 5343) and returns; algebraic rules silently not processed.

### pysces

Detected via `rule.isAlgebraic()` (line 1988); entry added to `__Errors__` (lines 1990–1996) and reported but does not halt import. **Rule is not stored or executed** — silently skipped after logging.

### SBMLToolkit.jl

Converted to algebraic constraint `0 ~ rhs` (line 8 in `rules.jl`). Included in ODESystem as implicit DAE constraint; ModelingToolkit's solver handles it. Species in AlgebraicRules with zero net stoichiometry flagged as `isbcspecies=true` to prevent conservation law reduction (line 62–72 in `utils.jl`). Limited by MTK's DAE capabilities — complex nonlinear systems may fail at solve time.

### Divergences

| ID   | Flag            | Library    | Description                                                                                                                                                                      |
| ---- | --------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D6   | `SPEC_SILENT`   | pysbml     | Spec requires exactly one undetermined variable but does not specify the algorithm. pysbml uses `sympy.solve`, which may fail for nonlinear rules or produce multiple solutions. |
| RR-A | `SPEC_CONFLICT` | roadrunner | LLVM backend throws `"Unable to support algebraic rules"`. Spec §4.8 requires support.                                                                                           |
| CP-A | `SPEC_CONFLICT` | copasi     | Sets `mUnsupportedRuleFound=true` and returns; algebraic rules silently not processed. Spec §4.8 requires support.                                                               |
| PS-C | `SPEC_CONFLICT` | pysces     | Added to `__Errors__` and skipped; not stored or executed. Spec §4.9.2 requires support.                                                                                         |

---

## 10. Constraint (§4.10)

### Spec

- Boolean math expression that should evaluate to `true` at all valid times.
- Optional `message` child: XHTML content for human-readable error.
- Violation is model-defined undefined behavior; interpreters **may** warn or halt.
- Constraints have no mathematical effect on the model.

### pysbml

```
LOGGER.warning("Constraints are not modelled")
# No further action.
```

### roadrunner

Behavior not determined from source review.

### copasi

`MCSBML + 49` warning issued at import; no further processing.

### pysces

`getNumConstraints()` detected; `"PySCeS does not support Constraints"` appended to fatal error list (line 1369). Model import terminates.

### SBMLToolkit.jl

`"listOfConstraints" in xml` → `throw(ErrorException("SBML models with listOfConstraints are not yet implemented."))` (line 270 in `systems.jl`). Model rejected entirely at parse time.

### Divergences

| ID   | Flag             | Library        | Description                                                                                                                             |
| ---- | ---------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| D7   | `SPEC_EXTENSION` | pysbml         | Constraints parsed but silently ignored. Spec says interpreter *may* warn or halt on violation; pysbml issues no warning at solve time. |
| CP-D | `SPEC_EXTENSION` | copasi         | Constraints warned about at import time but not processed or checked at solve time.                                                     |
| PS-B | `SPEC_CONFLICT`  | pysces         | Models with constraints are rejected entirely at import. Spec does not require rejecting models that contain constraints.               |
| JL-A | `SPEC_CONFLICT`  | SBMLToolkit.jl | Throws ErrorException for any model with `listOfConstraints`. Spec allows warnings but does not require model rejection at parse time.  |

---

## 11. Reaction + KineticLaw (§4.11)

### Spec

**Reaction attributes:**

| Attribute    | Required | Default | Meaning                                              |
| ------------ | -------- | ------- | ---------------------------------------------------- |
| `reversible` | yes      | true    | Informational only in L3v2; no mathematical effect   |
| `fast`       | —        | —       | **Removed in L3v2.** Presence is undefined behavior. |

**KineticLaw:** `math` gives the reaction rate in units of **extent/time** (§4.11.7). Local parameters scoped to kinetic law; shadow global symbols.

**Rate of change formula (§4.11.7):**

```
dn_S/dt = convFactor_S · Σ_j (stoich_{S,Rj} · v_{Rj})
```

### pysbml

**`rxn_had_compartment` pre-computation** [SPEC_SILENT D4]: detects compartment symbol in kinetic law → removes it and multiplies stoichiometry by compartment.

**`apply_conversion_factors`**: multiplies stoichiometry by conversion factor symbols.

**`convert_fast_reactions`** [SPEC_EXTENSION D3/D11]: QSS reduction; deferred-QSS event injection.

### roadrunner

D4 heuristic not needed (HOSU loading cancels compartment). Optional `CONSERVED_MOIETIES` flag (RR-B). `EvalConversionFactorCodeGen`, `EvalVolatileStoichCodeGen`.

### copasi

All kinetic laws divided by vol at import (CP-F). HOSU=true species multiplied by vol in kinetic law. `fast=true` → normal reaction + flag in `mFastReactions` (CP-E).

### pysces

Kinetic laws extracted via `getKineticLaw().getMath()` (line 1744) and converted to infix. **No compartment factor normalization** — rates assumed pre-scaled. Local parameters hashed with reaction name (e.g., `rxnA_k1`, line 1782). **`fast=true` ignored** with warning (lines 1737–1739, 1887–1893). Conversion factors not implemented. `stoichiometryMath` rejected (lines 1831–1839).

### SBMLToolkit.jl

`SBML.extensive_kinetic_math()` (line 8 in `reactions.jl`) converts concentration-based laws to extent-based by dividing HOSU=false species references by compartment. Local parameters promoted to global during preprocessing (`convert_promotelocals_expandfuns`). **Reversible reactions** (line 12–23 in `reactions.jl`): kinetic law split into forward and reverse components via symbolic expansion; warning issued if split is ambiguous. `fast=true` silently treated as normal reaction. Conversion factors not implemented.

### Divergences

| ID   | Flag             | Library        | Description                                                                                                                                                                                   |
| ---- | ---------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D3   | `SPEC_EXTENSION` | pysbml         | `fast=True` triggers full QSS reduction. L3v2 spec §4.11 explicitly removed `fast`.                                                                                                           |
| D4   | `SPEC_SILENT`    | pysbml         | `rxn_had_compartment` heuristic: auto-detects and auto-corrects traditional concentration/time kinetic laws. Not needed by roadrunner or SBMLToolkit.jl. **Scope gap:** for multi-compartment reactions where only one compartment appears in the kinetic law, all species stoichiometries are multiplied by their respective compartments, producing a spurious volume factor for species in other compartments. Not tested by the 833 D4 cases. |
| D11  | `SPEC_EXTENSION` | pysbml         | Deferred QSS: fast reaction inactive at t=0 → species kept as state variable; event assignments injected to snap to QSS.                                                                      |
| RR-B | `SPEC_EXTENSION` | roadrunner     | Optional conservation law reduction via L0-matrix null-space analysis. Introduces `CM_*` parameters not in original SBML.                                                                     |
| CP-E | `SPEC_EXTENSION` | copasi         | `fast=true` converted to normal reaction; no QSS semantics.                                                                                                                                   |
| CP-F | `SPEC_SILENT`    | copasi         | All kinetic laws divided by compartment vol at import. Correct for spec-compliant extent/time laws but wrong for traditional concentration/time laws without an explicit volume factor.       |
| JL-E | `SPEC_SILENT`    | SBMLToolkit.jl | Reversible kinetic laws split into forward/reverse via symbolic expansion; may produce ambiguous or incorrect splits for non-standard rate law forms. Warning issued when split is ambiguous. |

---

## 12. Event (§4.12)

### Spec

**Structure:**

```
Event
  useValuesFromTriggerTime: boolean  (required, no default)
  Trigger (initialValue, persistent, math)
  Priority (optional)
  Delay (optional)
  ListOfEventAssignments
```

**Trigger:** fires on **false → true** transition. `initialValue=true` → cannot fire at t=0. `persistent=false` → cancelled if trigger goes false before delay expires.

**`useValuesFromTriggerTime`:** `true` → math evaluated at trigger time, applied at execution. `false` → math evaluated at execution time.

**Priority:** higher executes first; equal → random order; re-evaluated after each execution.

**Delay:** evaluated at trigger time; execution time = trigger + delay.

**EventAssignment:** sets species quantity (amount if HOSU=true, concentration if HOSU=false).

### pysbml

Full implementation: trigger, delay, priority, useValuesFromTriggerTime, persistent all stored. `substitute_delays` (D10) and `substitute_rate_of` (D9). EventAssignment to HOSU=false: multiply by compartment (D8). `_handle_conc_boundary` conservation (D12). Test simulator in `tests/test_import.py` with LSODA, pending queue, random priority shuffle.

### roadrunner

EventAssignment to HOSU=false multiplied by compartment at store (identical to D8). rateOf fully implemented. delay throws (RR-C).

### copasi

`importEvent()` (line 7153). initialValue → `fireAtInitialTime`; persistent → `persistentTrigger`; priority supported; `useValuesFromTriggerTime` → `delayAssignment`. HOSU=true EventAssignment divided by vol (lines 7422–7445). delay/rateOf via auxiliary parameters (CP-B, CP-C).

### pysces

Events parsed with trigger, persistent flag, priority, and delay (lines 1432–1493). Hard constraints enforced:
- `trigger.initialValue=false` → `NotImplementedError` raised (lines 1439–1445)
- `useValuesFromTriggerTime=false` → `NotImplementedError` raised (lines 1446–1452)

EventAssignment stored as raw variable→formula pairs; no amount/concentration conversion applied. Delays supported and stored (lines 1471–1475).

### SBMLToolkit.jl

Events converted to `ContinuousVectorCallback` pairs (line 7 in `events.jl`): `[trigger_eq] => [effect_eqs]`. Passed to `ReactionSystem(..., continuous_events=cevs)` (line 146 in `systems.jl`). EventAssignment to HOSU=false: multiplied by compartment (line 21). Parameters not in reactions given `D(var) ~ 0` (line 96–102 in `systems.jl`). **`initialValue` and `persistent` not handled** — events always fire on upward crossing (line 36 warning). **`useValuesFromTriggerTime` not handled** — current values always used. **Delay throws** (JL-B). **Priority throws** (JL-C).

### Divergences

| ID   | Flag             | Library        | Description                                                                                                                                                                                                                                    |
| ---- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D8   | `SPEC_SILENT`    | pysbml         | EventAssignment to HOSU=false: multiply by compartment. Spec §4.12.5 does not specify the conversion. roadrunner does the same; SBMLToolkit.jl does the same; pysces does not convert; copasi divides by vol for HOSU=true (mirror direction). **Timing sub-case:** with `useValuesFromTriggerTime=true` + delay + dynamic compartment, C is frozen at trigger time → `S_conc = x_trigger × C_trigger / C_execution` ≠ `x_trigger`. Cases 01507/01508/01511. Spec silent. |
| D9   | `SPEC_SILENT`    | pysbml         | `rateOf` csymbol (L3v2 package, not Core): silently accepted. pysbml via sympy substitution; roadrunner via LLVM codegen; SBMLToolkit.jl native (`D()`); copasi via auxiliary parameter; pysces: not implemented.                              |
| D10  | `SPEC_EXTENSION` | pysbml         | `delay(x, d)` in kinetic laws / derived rules: time-shift substitution. Spec defines Delay only within Events. True DDEs raise `NotImplementedError`.                                                                                          |
| D12  | `SPEC_SILENT`    | pysbml         | `_handle_conc_boundary`: compartment EventAssignment without species assignment → conserves amount via `{k}_conc_new = {k}_conc_old * C_old / C_new`.                                                                                          |
| RR-C | `SPEC_CONFLICT`  | roadrunner     | `delay` csymbol always throws. Spec §3.4.6 defines delay as a valid MathML operator.                                                                                                                                                           |
| CP-B | `SPEC_SILENT`    | copasi         | `delay()` converted to auxiliary global parameter + assignment rule workaround.                                                                                                                                                                |
| CP-C | `SPEC_SILENT`    | copasi         | `rateOf()` converted to auxiliary global parameter `rateOf_<id>` + assignment rule.                                                                                                                                                            |
| PS-D | `SPEC_EXTENSION` | pysces         | `trigger.initialValue=false` raises `NotImplementedError`. `useValuesFromTriggerTime=false` raises `NotImplementedError`. Both are required spec features.                                                                                     |
| PS-E | `SPEC_EXTENSION` | pysces         | `delay()` csymbol stripped from kinetic law; kinetic law added to `delayignore` list. No approximation or substitution — the delay is simply removed.                                                                                          |
| JL-B | `SPEC_CONFLICT`  | SBMLToolkit.jl | Delay in events throws `ErrorException`. Spec §4.12.4 defines delays as a required event feature.                                                                                                                                              |
| JL-C | `SPEC_CONFLICT`  | SBMLToolkit.jl | Event priority throws `ErrorException`. Spec §4.12.3 defines priority as a valid event feature.                                                                                                                                                |
| JL-D | `SPEC_SILENT`    | SBMLToolkit.jl | `trigger.initialValue` and `persistent` semantics not implemented; events always fire on upward crossing without initialValue consideration. Warning issued at line 36 in `events.jl`.                                                         |

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

Use the `sbml-add-library` skill (`/sbml-add-library`). It handles source discovery,
Explore-subagent delegation, content writing, and git commit.

Manual steps: add a `### LibraryName` subsection inside each element section (before
`### Divergences`), add a column to the **Feature Matrix**, add a **Library Architectures**
paragraph, and add library-specific divergence IDs (e.g. `XX-A`) to Divergences tables.
