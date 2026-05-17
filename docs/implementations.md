# SBML Implementation Comparison

Tracks how each tool transforms SBML L3v2 into an executable ODE system.
Organized by SBML element; each section has a `### LibraryName` subsection per tool.
Add a new `### LibraryName` block inside each element when comparing more tools.

**Spec:** SBML Level 3 Version 2 Core Release 2 (29 March 2019)
**Divergence flags:** See [`divergences.md`](divergences.md) for flag definitions and the full test-case table.

---

## Feature Matrix

| Feature                         | pysbml                                             | roadrunner                                           | copasi                                                        | pysces                                                                     | SBMLToolkit.jl                                                    | amici                                                                              | SBMLToolbox                                                                                   | sbscl                                                                                         | morpheus                                                                             | vcell                                                                                            |
| ------------------------------- | -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Internal species storage        | Dual: explicit `{k}_amount` + `{k}_conc` variables | Amount-canonical; concentration derived at read time | Concentration-canonical; amount derived as `conc × vol`       | Mode-switched: conc unless ANY HOSU=true → amount-canonical for all (PS-F) | Amount-canonical; initialConc converted at import                 | Dual: concentration by default (dC/dt); HOSU=true → amount-canonical (dA/dt)      | Amount in ODE state vector; `isConcentration` flag selects output scaling             | Flexible dual: `isAmount[]` per-species; conversion at evaluation time in SpeciesValue        | Dual: amount internal; concentration derived as `amount/vol` in expressions          | Concentration-canonical; initialAmount → conc at import (÷ compartment); no explicit amount var |
| HOSU effect                     | Selects one of ~8 `_handle_*` dispatch paths       | Affects load/store conversion only                   | HOSU=true → multiply by vol in kinetic laws, divide elsewhere | Triggers model-wide amount mode if any species has HOSU=true               | HOSU=false → divide by vol in kinetic law; HOSU=true → raw amount | `"amount": HOSU` flag selects ODE type; `_transform_dxdt_to_concentration()` for non-HOSU | `hasAmountOnly=1` → raw amount; else KL divided by compartment in ODE output (ST-C)  | SpeciesValue converts at eval time based on HOSU × isAmount flag combination                  | HOSU=true → formula_symbol = species name; else formula_symbol = `amount/vol`        | **Not branched at import** (VC-A); all species handled uniformly; HOSU stored but unused        |
| Algebraic rules                 | `sympy.solve` → assignment rule (D6)               | **Throws** (RR-A)                                    | **Silently ignores** (CP-A)                                   | Silently ignores with error log (PS-C)                                     | Algebraic constraint `0 ~ rhs` (MTK DAE)                          | Supported (L3+ only; L2 raises SBMLException) via SUNDIALS DAE                    | Converted to AssignmentRule if analytically isolatable; **throws** if not (ST-B)    | Converted to assignment via AlgebraicRuleConverter; throws if overdetermined (SC-A)           | **Throws** SBMLConverterException (MO-A)                                             | Silently ignored with warning "not handled at this time" (VC-B)                                 |
| Conservation law reduction      | None                                               | Optional L0-matrix moiety analysis (RR-B)            | None                                                          | None                                                                       | None                                                              | Automatic optional; default enabled; L0-matrix approach (AM-D)                    | None                                                                                  | None                                                                                          | None                                                                                 | None                                                                                             |
| Kinetic law compartment factor  | D4 auto-strip heuristic                            | None needed (HOSU loading cancels compartment)       | Divides ALL kinetic laws by vol at import (CP-F)              | None (assumes rates already correct)                                       | HOSU=false species divided by vol (`extensive_kinetic_math`)      | Division by compartment in `dx_dt` in de_export.py; no explicit heuristic needed  | KL divided by compartment for `isConcentration=1` species in ODE output (ST-C)      | KL divided by compartment in processVelocities for concentration-canonical species            | Conversion factors applied; no explicit KL/compartment division                      | Deferred to GeneralLumpedKinetics/codegen layer; no explicit D4-style heuristic at import       |
| EventAssignment to HOSU=false   | Multiply by compartment (D8)                       | Multiply by compartment at store                     | Direct assignment (concentration is native)                   | No conversion applied                                                      | Multiply by compartment                                           | Direct concentration assignment (native; concentration is canonical state)         | Direct assignment; no concentration→amount conversion (ST-A)                         | Direct Y assignment; compartment change triggers species concentration rebalance               | Multiply by compartment (lines 1131–1132)                                            | Direct assignment; no HOSU-based conversion (VC-I)                                              |
| EventAssignment to HOSU=true    | Direct (amount is native)                          | Direct (amount is native)                            | Divide by compartment (amount → concentration)                | No conversion applied                                                      | Direct (amount is native)                                         | Direct amount assignment (native; amount is canonical state for HOSU=true)         | Direct amount assignment                                                              | Direct amount assignment                                                                      | Direct amount assignment                                                             | Direct assignment                                                                                |
| Rate rule + dynamic compartment | Product rule correction (D5)                       | Same product rule                                    | Not documented                                                | No product rule                                                            | Product rule applied (D(S) ~ C·f + S/C·D(C))                      | `_transform_dxdt_to_concentration()` applies product rule for non-HOSU species     | No product rule                                                                       | Product rule via changeRate correction (RateRuleValue.java line 129)                          | Post-processing adjusts changeRate for compartment species (lines 671–713)           | Product rule not explicitly applied at import (VC-F)                                            |
| rateOf csymbol                  | Sentinel + sympy substitution (D9)                 | Native LLVM IR codegen with quotient-rule            | Auxiliary parameter workaround (CP-C)                         | Not implemented                                                            | Native: `rateOf` → `D()` (MTK derivative)                         | Native support via `_process_sbml_rate_of()` and `_rateof_to_dummy()`             | Not implemented (undefined function call)                                             | Native via ASTNodeInterpreter.rateOf(); reads changeRate[] or rate rule RHS                   | Symbol renaming: `rateOf(X)` → `X.rate` (L3v2 only)                                 | **Not supported**; no code found; likely throws at ODE generation (VC-E)                        |
| delay csymbol                   | Time-shift approximation (D10)                     | **Throws** (RR-C)                                    | Auxiliary parameter workaround (CP-B)                         | Stripped from kinetic law, not replaced (PS-E)                             | **Throws** (JL-B)                                                 | **Throws** SBMLException for non-zero delays (AM-A)                               | **Throws** "Cannot deal with delayed events" (ST-D)                                  | Native via DelayValueHolder; evaluates at t−delay                                             | Auxiliary DelayProperty/DelayVariable elements (MO-B)                                | Warning logged; expression **replaced with 0.0**; model continues with broken expression (VC-D) |
| No initialAmount/initialConc    | 0.0 + amount-vs-conc heuristic (D1)                | 0.0 + `LOG_WARNING`                                  | Unknown                                                       | 0.0 (libsbml default, silent)                                              | 0 (SBML.jl default)                                               | 0.0 silent default (`get_species_initial()` line 3434)                            | **Throws** "species concentration not provided or assigned by rule" (ST-E)           | Majority-vote default 0.0 (determineMajorSpeciesAttributes); silent                           | Default 0.0; silent                                                                  | Silently defaults to 0.0                                                                        |
| Constraints                     | Silently ignored (D7)                              | Unknown                                              | Warns and ignores (CP-D)                                      | Warns + sleeps; model still loads (PS-B)                                   | **Throws** (JL-A)                                                 | Silently ignored; no code found for `getListOfConstraints()`                      | **Throws** "Cannot deal with constraints." (ST-F)                                    | Evaluated every step via listener; violations logged; simulation continues                    | Silently dropped with log message (MO-C)                                             | **Throws** SBMLImportException (VC-C)                                                           |
| InitialAssignment               | Supported                                          | Supported                                            | Supported                                                     | Warns + sleeps; model still loads (PS-A)                                   | Supported                                                         | Supported                                                                          | Supported; evaluated at t=0                                                           | Supported; iterative evaluation until Y stabilizes                                            | Supported; applied during element processing                                         | Supported; two-phase parse + apply after all objects created                                    |
| fast=true reaction              | QSS reduction + deferred events (D3/D11)           | Unknown                                              | Converted to normal reaction, no QSS (CP-E)                   | Ignored with warning                                                       | Silently treated as normal reaction                               | **Throws** SBMLException (AM-C)                                                   | **Throws** "Cannot deal with fast reactions" (ST-G)                                  | Supported via separate fast-reaction solver phase                                             | Silently treated as normal reaction                                                  | Supported via annotation flag; marked for QSS treatment                                         |
| Conversion factors              | Applied to stoichiometry                           | EvalConversionFactorCodeGen                          | Unknown                                                       | Not implemented                                                            | Not implemented                                                   | Applied to stoichiometry (line 1389); model-level and species-level               | **Throws** "Cannot deal with conversion factors" (ST-H)                              | Applied via conversionFactors[] per species                                                   | Applied to stoichiometry                                                             | Applied to stoichiometry (line 1250)                                                            |
| Dynamic stoichiometry           | SpeciesRef → parameter                             | EvalVolatileStoichCodeGen                            | Stoichiometric expression map                                 | stoichiometryMath rejected                                                 | Local params promoted to global                                   | `_get_list_of_species_references()` (line 3437); SpeciesRef handled               | stoichiometryMath converted to formula                                                | Via JSBML stoichiometryMath handling                                                          | stoichiometryMath converted to formula                                               | stoichiometryMath parsed as expression (lines 1110–1118 for reactants)                          |

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

**amici** — `ref/amici/python/sdist/amici/importers/sbml/__init__.py` (Python, C++ backend; previously `sbml_import.py` — line numbers in subsections below may have shifted after restructuring)
SBML → SUNDIALS ODE/DAE C++ code via libsbml symbolic parsing. Dual per-species canonical form (HOSU flag selects amount vs concentration state); compartment volume factors deferred to C++ codegen (`de_export.py`). Conservation law reduction default-enabled. Events handled by SUNDIALS root-finding; delays and non-persistent triggers not supported.

**SBMLToolbox** — `ref/SBMLToolbox/src/Simulation/` (MATLAB)
SBML → MATLAB ODE function files (`WriteODEFunction.m`, `WriteEventHandlerFunction.m`, `WriteEventAssignmentFunction.m`). Single-compartment only; amounts in state vector with concentration derived at output. Many unsupported features (delays, conversion factors, multiple compartments, constraints) throw explicit MATLAB errors.

**sbscl** — `ref/sbscl/src/main/java/org/simulator/sbml/` (Java)
SBML → interpreted ODE system via `SBMLinterpreter.java` + `EquationSystem.java`. Flexible per-species canonical form (`isAmount[]` array); full event support including delays and priority; algebraic rules converted via `AlgebraicRuleConverter`; constraints evaluated via listener pattern. Native delay() and rateOf() support.

**morpheus** — `ref/morpheus/gui/sbml_converter.cpp` (C++)
SBML → MorpheusML XML format via 1428-line converter. Designed for spatial cell-based modeling; SBML import is best-effort translation. Algebraic rules throw; event priorities dropped; delay() and rateOf() supported via auxiliary variable workarounds.

**vcell** — `ref/vcell/vcell-core/src/main/java/org/vcell/sbml/vcell/SBMLImporter.java` (Java)
SBML → VCell BioModel/SimulationContext objects via 4581-line multi-pass importer. Concentration-canonical; all species handled uniformly without HOSU branching. Annotation-rich design preserves VCell-specific metadata. Algebraic rules and rateOf not supported; delay() replaced with 0.0 workaround.

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

### amici

All FunctionDefinitions are preprocessed via `SBMLFunctionDefinitionConverter()` before the main import loop (lines 249–253 in `importers/sbml/__init__.py`). Functions are inlined into kinetic laws and rules at the libsbml level before any symbolic parsing begins. No function objects are created or stored in the generated C++ code.

### SBMLToolbox

Functions parsed and emitted as nested MATLAB functions at the end of the generated ODE file (WriteODEFunction.m lines 530–553). Arguments extracted from the lambda expression; body written as a named MATLAB function. No recursion detection or argument count validation; inlining is purely textual substitution.

### sbscl

`FunctionValue.computeDoubleValue()` (astnode/FunctionValue.java lines 45–163) evaluates function calls by mapping argument values into `argumentValues[]` (line 91) and recursively evaluating the body with the index mapping. Proper per-call scoping; no global function-object storage.

### morpheus

FunctionDefinitions converted to MorpheusML `<Function>` elements (lines 940–959 in sbml_converter.cpp). Body expression converted via `formulaToString()`. Inlining code at lines 978–980 is commented out, suggesting potential limitations with recursive function replacement.

### vcell

Functions inlined at parse time via `LambdaFunction[]` array (`addFunctionDefinitions()`, lines 526–574 in SBMLImporter.java). Arguments renamed with unique suffix to avoid symbol collision during substitution. All FunctionDefinition identifiers are substituted into kinetic laws and rule expressions before model assembly. No divergences from spec.

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

### amici

Unit parsing disabled entirely via `setParseUnits(L3P_NO_UNITS)` on the libsbml reader (lines 195–197 in `importers/sbml/__init__.py`). UnitDefinitions are not extracted, stored, or validated. No runtime unit enforcement.

### SBMLToolbox

Units completely ignored during ODE generation. No unit definitions are extracted, enforced, or validated; no unit-driven species scaling occurs.

### sbscl

Units annotation-only (SBMLinterpreter.java constructor docstring, line 86: "Note that currently, units are not considered."). Unit information in ASTNodes is stored but ignored during evaluation.

### morpheus

Units discarded via recursive `removeUnits()` helper (lines 60–67 in sbml_converter.cpp), called before every math-to-string conversion. Import dialog explicitly warns "Units are discarded." No enforcement.

### vcell

Unit definitions extracted and stored in `sbmlUnitIdentifierHash` (`createSBMLUnitSystemForVCModel()`, lines 2068–2182). Predefined SBML unit names pre-populated (lines 2086–2149). Used for dimensionality tracking only; no runtime enforcement or dimensional analysis.

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

### amici

Compartments stored as scalar symbolic values (lines 1088–1106 in `importers/sbml/__init__.py`). Constant compartments become C++ parameter entries. Non-constant compartments governed by RateRule are promoted to AMICI state variables (species-like objects). Volume scaling in `d(amount)/dt = v/C` is applied in the C++ code generation layer (`de_export.py`), not at SBML import time.

### SBMLToolbox

**Single-compartment only**: throws MATLAB error if the model has more than one compartment (WriteODEFunction.m line 107). **Constant compartments only**: throws if constant=false (line 78). Compartment size written as a numeric constant in the generated ODE file (line 253); used only to convert amounts to concentrations for species output.

### sbscl

Compartments stored in the Y vector before species entries (EquationSystem.java lines 500–511) with `constantHash` tracking mutability. Dynamic compartments supported via rate rules. When compartment size changes via an assignment rule, `updateSpeciesConcentrationByCompartmentChange()` (lines 848–863) rescales all contained non-amount species: `Y[s] = Y[s] * oldSize / newSize`.

### morpheus

Constant compartments → MorpheusML `<Constant>`; dynamic (constant=false) → `<Variable>` (lines 627–661). InitialAssignments and AssignmentRules applied to compartment value (lines 635–640). Dynamic compartments trigger post-processing compensation logic that adjusts species ODE rates for volume changes (lines 671–713).

### vcell

Two-pass handler in `addCompartments()` (lines 230–396): first pass creates VCell Structure objects (Feature/Membrane); second pass parses compartment size expressions. Dynamic compartments supported if a RateRule targets the compartment ID (line 382). Outside topology resolved via annotation or `outside` attribute (lines 291–340).

### Divergences

| ID   | Flag             | Library     | Description                                                                                                                               |
| ---- | ---------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| PS-G | `SPEC_SILENT`    | pysces      | COMP_FUDGE_FACTOR rescales compartments smaller than 1e-6. Non-standard heuristic not described in spec; affects initial species amounts. |
| ST-I | `SPEC_CONFLICT`  | SBMLToolbox | Single-compartment models only; models with multiple compartments throw a MATLAB error. Spec §4.5 requires support for multiple compartments. |

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

### amici

Dual per-species canonical form selected by HOSU flag (lines 1108–1182 in `importers/sbml/__init__.py`). `_get_species_initial()` (line ~3289–3316) resolves `initialAmount` vs `initialConcentration`, converting between them using compartment size. HOSU=false → concentration-canonical ODE (`dC/dt`), amount derived as `A = C × vol`. HOSU=true → amount-canonical ODE (`dA/dt`), concentration derived as `C = A / vol`. `_transform_dxdt_to_concentration()` applies the product rule for HOSU=false species in dynamic compartments. `boundaryCondition=true` → species excluded from ODE state vector. Neither initialAmount nor initialConcentration → 0.0 silent default.

### SBMLToolbox

Amount in ODE state vector; `isConcentration` flag computed in AnalyseSpecies.m (lines 143–177) based on which initial value is set. HOSU=true (`hasAmountOnly=1`) → amount-canonical ODE; else KL divided by compartment at output. `boundaryCondition=true` → reaction rate returns '0' (GetRateLawsFromReactions.m line 80). Neither initialAmount nor initialConc → **throws** "species concentration not provided or assigned by rule". EventAssignment direct assignment without concentration→amount conversion (ST-A).

### sbscl

Per-species flexible dual via `isAmount[]` array (EquationSystem.java lines 544–569). `SpeciesValue.computeDoubleValue()` (lines 122–173) converts based on four HOSU × isAmount combinations: amount+notHOSU → divide by C; notAmount+HOSU → multiply by C; else identity. Neither initial value → `determineMajorSpeciesAttributes()` majority-vote default 0.0 (lines 1267–1291). `boundaryCondition=true` → `zeroChange=true` (lines 791–796), excluded from ODE stoichiometry. EventAssignment to compartment triggers `updateSpeciesConcentrationByCompartmentChange()`.

### morpheus

Dual: amount stored internally; concentration derived as formula expression (`addSBMLSpecies()`, lines 810–898). HOSU=true → formula_symbol = species name (raw amount); else formula_symbol = `amount/vol`. `initialConcentration` takes priority over `initialAmount` (line 852); neither → default 0. `boundaryCondition=true` → ODE generation skipped (lines 1294, 1371). EventAssignment to amount-canonical species: multiply by compartment (lines 1131–1132).

### vcell

Concentration-canonical: `addSpecies()` (line 1551) converts `initialAmount` → concentration by dividing by compartment size at import. HOSU **not branched** at import (VC-A): all species stored as `SpeciesContext` uniformly. `boundaryCondition=true` → species clamped (line 1628). Neither initial value → silently defaults to 0.0. EventAssignment direct assignment without HOSU-based conversion (VC-I).

### Divergences

| ID   | Flag            | Library     | Description                                                                                                                                                                                                                                                                                                        |
| ---- | --------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| D1   | `SPEC_SILENT`   | pysbml      | Species with neither `initialAmount` nor `initialConcentration`: spec says value is "unknown or from external source" (§4.6.4). pysbml injects 0.0 using a co-reactant heuristic to guess amount vs concentration. Tests: t676, t688 → amount; t1513 → concentration.                                              |
| D2   | `SPEC_SILENT`   | pysbml      | `_handle_amount_boundary_has_substance_units` (test 1123): source FIXME — despite `hasOnlySubstanceUnits=True`, derives `{k}_conc` and forces `rxn_had_compartment=True`. Spec does not specify this interaction.                                                                                                  |
| D13  | ~~`SPEC_CONFLICT`~~ **FIXED** | pysbml | `constant=true`, `hasOnlySubstanceUnits=false`, initialized via `initialConcentration` in a non-constant compartment. **Fixed:** now stores `{k}_amount = initConc × C(0)` as constant parameter and derives `k = {k}_amount / C(t)`. Cases 01117, 01118 pass. |
| PS-F | `SPEC_SILENT`   | pysces      | Model-wide HOSU mode: if ANY species has `hasOnlySubstanceUnits=true`, ALL species switch to amount-canonical. Spec defines HOSU per-species; pysces treats it as a global model flag.                                                                                                                             |
| ST-A | `SPEC_SILENT`   | SBMLToolbox | EventAssignment to HOSU=false species: direct assignment without concentration→amount conversion. Same as pysces; contrast pysbml/roadrunner which multiply by compartment.                                                                                                                                         |
| ST-E | `SPEC_CONFLICT` | SBMLToolbox | Species with neither `initialAmount` nor `initialConcentration` throws MATLAB error "species concentration not provided or assigned by rule". Spec §4.6.4 says value is "unknown"; tools should default to 0.0 or use heuristics, not abort.                                                                       |
| VC-A | `SPEC_SILENT`   | vcell       | HOSU flag not branched at import; all species handled uniformly as SpeciesContext objects. Species that have `hasOnlySubstanceUnits=true` receive no special treatment during ODE generation. Spec §4.6.2 requires HOSU to affect how species identifiers appear in formulas.                                       |

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

### amici

Parameters classified into four categories (lines 1306–1420 in `importers/sbml/__init__.py`): `Fixed` (constant=true, no rule) → C++ `p[]` array; `Dynamic` (constant=false with RateRule) → state variable; `Expression` (constant=false with AssignmentRule) → observable/derived expression; parameter governed by EventAssignment → promoted to state variable. Constant parameters without any rule become C++ compile-time constants.

### SBMLToolbox

Constant parameters → literal numeric values in ODE file (WriteODEFunction.m line 261). Non-constant parameters added to ODE state vector alongside species (lines 237, 300–301). Event assignments to non-constant parameters allowed.

### sbscl

Parameters stored in Y vector (EquationSystem.java lines 593–604) with `constantHash` tracking mutability. Local parameters of kinetic laws updated via `KineticLaw.getMath().updateVariables()` (line 628). Event-assigned parameters allowed.

### morpheus

Constant → MorpheusML `<Constant>` (line 913 in sbml_converter.cpp); constant=false → `<Variable>` or `<Property>`. InitialAssignments and AssignmentRules applied to parameter value (lines 925–929). Local reaction parameters renamed with scheme `{param}_{reaction_number}` if conflicting with globals (lines 1187–1221).

### vcell

`ModelParameter` objects created in `addParameters()` (lines 586–950 in SBMLImporter.java). Expressions resolved in deferred two-phase pattern. Constant parameters immutable at runtime except via event assignments. Spatial-model parameters (DiffusionCoefficient, AdvectionCoefficient, BoundaryCondition) mapped to `SpeciesContextSpec` entries rather than ModelParameter.

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

### amici

Topologically sorted and evaluated at t=0 (lines 2259–2296 in `importers/sbml/__init__.py`). May reference parameters, compartments, and species. Overrides initial amounts/concentrations and stoichiometric SpeciesReference values. Evaluation order determined by dependency analysis before C++ code generation.

### SBMLToolbox

Evaluated at t=0 via `Substitute()` on the SBML math expression (GetSpecies.m lines 97–110). Result replaces the species/parameter initial value. Applied before AssignmentRules. No iterative re-evaluation for cyclic dependencies.

### sbscl

Iterative evaluation loop in `SBMLinterpreter.init()` (lines 447–490) runs until the Y vector stabilizes. Handles cyclic dependencies between initial assignments and assignment rules. Targets: species, parameters, compartments, stoichiometric SpeciesReferences.

### morpheus

Applied inline during species/compartment/parameter element processing (lines 883–895 for species, 635–640 for compartments, 924–929 for parameters). Note: dedicated `addSBMLInitialAssignments()` function (lines 1398–1414) exists but is never called; all InitialAssignment handling is inline.

### vcell

Two-phase: parse expressions in `parseAssignmentAndInitialAssignmentExpressions()` (lines 1288–1316), apply to target entries after all objects are created (lines 3023–3037). Handles species, parameters, compartment sizes, and SpeciesReference stoichiometry.

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

### amici

Converted to observable or algebraic expressions (lines 1662–1704 in `importers/sbml/__init__.py`). Targets removed from the ODE state vector; their values computed from the rule expression at every time step. HOSU handling follows the same per-species amount/concentration classification as reaction species. Stoichiometric SpeciesReference targets of assignment rules are also supported.

### SBMLToolbox

Written as explicit formula lines in the ODE file (WriteODEFunction.m lines 305–314). For species with assignment rules and no reactions or rate rules, the rule derivative is computed via symbolic differentiation `DifferentiateRule` (lines 427–429) and used as the ODE rate.

### sbscl

`AssignmentRuleValue` objects evaluated in `processRules()` (SBMLinterpreter.java lines 636–695) in a loop for `numberOfAssignmentRulesLoops` iterations. Topological ordering applied in EquationSystem (lines 1033–1094). If rule targets a compartment, dependent species concentrations rebalanced (lines 675–681).

### morpheus

AssignmentRules → MorpheusML `<Equation>` elements (lines 1021–1046 in sbml_converter.cpp). For amount-canonical species with concentration-based rules, value multiplied by compartment (lines 1027–1029). Constant species with assignment rules skipped (lines 1011–1012).

### vcell

Two-phase: parse (lines 1295–1305), create VCell AssignmentRule and mark target species as clamped (lines 1340–1378 in SBMLImporter.java). Clamped species excluded from ODE state vector; governed entirely by assignment rule expression. StructureSize (compartment) assignment rules update compartment size and may adjust relative structure sizes (lines 1379–1388).

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

### amici

RateRule targets promoted to AMICI state variables with `dx_dt = rule_expr` (lines 1184–1270 in `importers/sbml/__init__.py`). `_transform_dxdt_to_concentration()` applies the product rule for HOSU=false species in dynamic compartments, equivalent to the D5 correction. Parameters and compartments governed by RateRules are similarly promoted to state variable entries in the C++ model.

### SBMLToolbox

Rate rule formulas become direct `d/dt` entries in ODE output (WriteODEFunction.m lines 409–416). No compartment volume division applied to rate rules. **No product rule for dynamic compartments.**

### sbscl

`RateRuleValue` objects set `changeRate[target]` directly to rule RHS (RateRuleValue.java lines 122–123). **Product rule applied for compartments**: if compartment has a rate rule and contains non-amount species, `changeRate[s] = -changeRate[c] * Y[s] / Y[c]` (line 129), correctly handling concentration changes from volume changes.

### morpheus

RateRules → MorpheusML `<DiffEqn>` elements (lines 1049–1065 in sbml_converter.cpp). Rate direction (amount or concentration) tracked in `diffeqn_map` (line 1064). Post-processing (lines 671–713) adjusts species rates for dynamic compartment volume changes.

### vcell

Two-phase: parse (lines 1404–1414), create VCell RateRule and mark target as clamped (lines 1449–1465 in SBMLImporter.java). Product rule for dynamic compartments **not explicitly applied** in import code (VC-F); may be deferred to the kinetics/codegen layer.

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

### amici

Supported for L3 models via SUNDIALS IDA DAE solver (lines 1528–1662 in `importers/sbml/__init__.py`). The algebraic rule `0 = f(...)` is added as a residual constraint to the DAE system. **L2 models**: raises `SBMLException("Algebraic rules are only supported for SBML L3.")`. The floating variable is identified symbolically but the constraint is solved numerically by SUNDIALS rather than analytically.

### SBMLToolbox

Converted to AssignmentRule if the equation can be analytically solved for one variable via `Rearrange()` (AnalyseSpecies.m lines 221–240). If the species also has reactions or rate rules, `ConvertedToAssignRule=0` and the rule is differentiated instead (WriteODEFunction.m line 452). Fails with MATLAB error if the equation cannot be isolated analytically.

### sbscl

`AlgebraicRuleConverter` (598 lines) converts algebraic rules to assignment rules using `OverdeterminationValidator` matching (EquationSystem.java line 1256). Throws `ModelOverdeterminedException` if system is overdetermined (line 1253). Successfully converted rules processed as assignment rules (lines 993–1025).

### morpheus

**Throws** `SBMLConverterException::SBML_ALGEBRAIC_RULE` immediately on encountering an algebraic rule (lines 1018–1020 in sbml_converter.cpp). No fallback or workaround.

### vcell

Silently ignored with VCLogger warning "Algebraic rules are not handled in the Virtual Cell at this time" (lines 2627–2636 in SBMLImporter.java). No exception thrown; model continues importing without the algebraic rule.

### Divergences

| ID   | Flag            | Library     | Description                                                                                                                                                                      |
| ---- | --------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D6   | `SPEC_SILENT`   | pysbml      | Spec requires exactly one undetermined variable but does not specify the algorithm. pysbml uses `sympy.solve`, which may fail for nonlinear rules or produce multiple solutions. |
| RR-A | `SPEC_CONFLICT` | roadrunner  | LLVM backend throws `"Unable to support algebraic rules"`. Spec §4.8 requires support.                                                                                           |
| CP-A | `SPEC_CONFLICT` | copasi      | Sets `mUnsupportedRuleFound=true` and returns; algebraic rules silently not processed. Spec §4.8 requires support.                                                               |
| PS-C | `SPEC_CONFLICT` | pysces      | Added to `__Errors__` dict with message "Algebraic rule (%s) ignored" (getRules() line 1990–1996). The rule formula IS parsed and stored in `__rules__` with `type='algebraic'`, but algebraic rules have no variable target and are not used in ODE generation. Spec §4.9.2 requires support. |
| ST-B | `SPEC_SILENT`   | SBMLToolbox | Converted to AssignmentRule only if analytically isolatable; throws MATLAB error if not. Partial support — equivalent of pysbml's `sympy.solve` fallback but without error recovery. |
| SC-A | `SPEC_SILENT`   | sbscl       | Throws `ModelOverdeterminedException` if the algebraic system is overdetermined. Spec §4.9.2 requires that an overdetermined system be detected and reported as an error, so this matches spec intent. |
| MO-A | `SPEC_CONFLICT` | morpheus    | Throws `SBMLConverterException` on any algebraic rule. Spec §4.9.2 requires support.                                                                                            |
| VC-B | `SPEC_SILENT`   | vcell       | Silently ignored with warning message. Same behavior as copasi (CP-A) and pysces (PS-C); spec §4.9.2 requires support.                                                           |

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

### amici

No code found for processing `getListOfConstraints()` in `importers/sbml/__init__.py` (confirmed by search). Constraints are silently ignored at import with no warning or error.

### SBMLToolbox

Throws MATLAB error "Cannot deal with constraints." (WriteODEFunction.m line 171) on any model that has constraints. ODE generation aborted immediately.

### sbscl

Constraints fully supported at runtime. `ConstraintEvent`/`ConstraintListener` pattern (EquationSystem.java lines 1991–2019): each constraint's boolean math evaluated every ODE step (line 1995); violation triggers `processViolation()` on registered listeners. `SimpleConstraintListener` auto-registered (line 710); logs violations but does not halt simulation.

### morpheus

Silently dropped with logged message "Dropped unsupported SBML Constraint..." (`parseMissingFeatures()`, lines 1417–1427 in sbml_converter.cpp). No exception thrown.

### vcell

Throws `SBMLImportException` "VCell doesn't support Constraints at this time" (`addConstraints()`, lines 510–514 in SBMLImporter.java). Model import aborted.

### Divergences

| ID   | Flag             | Library        | Description                                                                                                                             |
| ---- | ---------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| D7   | `SPEC_EXTENSION` | pysbml         | Constraints parsed but silently ignored. Spec says interpreter *may* warn or halt on violation; pysbml issues no warning at solve time. |
| CP-D | `SPEC_EXTENSION` | copasi         | Constraints warned about at import time but not processed or checked at solve time.                                                     |
| PS-B | `SPEC_CONFLICT`  | pysces         | Models with constraints are rejected entirely at import. Spec does not require rejecting models that contain constraints.               |
| JL-A | `SPEC_CONFLICT`  | SBMLToolkit.jl | Throws ErrorException for any model with `listOfConstraints`. Spec allows warnings but does not require model rejection at parse time.  |
| ST-F | `SPEC_CONFLICT`  | SBMLToolbox    | Throws MATLAB error for any model with constraints. Spec §4.10 allows warning or halting on violation; rejecting presence is stricter.  |
| MO-C | `SPEC_EXTENSION` | morpheus       | Constraints silently dropped with log message. No runtime evaluation. Same behavior as pysbml and amici.                               |
| VC-C | `SPEC_CONFLICT`  | vcell          | Throws SBMLImportException on presence of constraints. Same reasoning as JL-A / PS-B.                                                  |

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

### amici

Kinetic laws stored symbolically as-is; compartment volume scaling applied in the C++ code generation layer (`de_export.py`) rather than at import time (lines 1423–1493, 3027–3093 in `importers/sbml/__init__.py`). Local parameters globalized with uniquified names. `fast=true` raises `SBMLException` (AM-C). Conversion factors applied to stoichiometry at line 1389. Dynamic stoichiometry (stoichiometryMath / SpeciesReference rules) handled via `_get_list_of_species_references()` (line 3437). Conservation law reduction enabled by default via L0-matrix analysis (AM-D).

### SBMLToolbox

KL divided by compartment for `isConcentration=1` species in ODE output (WriteODEFunction.m line 399): `xdot = (KL)/compartment`. fast=true → **throws** "Cannot deal with fast reactions". Conversion factors → **throws** "Cannot deal with conversion factors". Local parameters renamed with reaction ID (GetRateLawsFromReactions.m lines 140–145). stoichiometryMath converted to formula (lines 171–198).

### sbscl

KL rate divided by compartment in `processVelocities()` for concentration-canonical (non-amount, HOSU=false) species (lines 767–769). fast=true reactions evaluated only when `isProcessingFastReactions=true` (line 739). Conversion factors applied via `conversionFactors[]` (line 770). rateOf() and delay() natively supported. Dynamic stoichiometry via JSBML stoichiometryMath.

### morpheus

Kinetic law converted via `formulaToString()`. Species conversion factors applied to stoichiometry (lines 1288–1290, 1364–1366 in sbml_converter.cpp). fast=true not checked — treated as normal. Local parameters renamed if conflicting with globals (lines 1187–1221). stoichiometryMath converted to formula (lines 1265, 1357). Dynamic SpeciesReference stoichiometry stored as Variable nodes (lines 1249–1252).

### vcell

KL parsed via `getExpressionFromFormula()` and adjusted (lines 3252–3253 in SBMLImporter.java). `GeneralLumpedKinetics` vs `GeneralKinetics` chosen by spatial flag. Compartment factor deferred to kinetics codegen layer. fast=true → annotation flag, marked for QSS treatment. Conversion factors applied to stoichiometry (line 1250). delay() → warning + 0.0 replacement (lines 2468–2473). rateOf() not supported (VC-E).

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
| AM-C | `SPEC_CONFLICT`  | amici          | `fast=true` raises `SBMLException`. L3v2 §4.11 does not define `fast`; per spec, presence is undefined behavior. AMICI treats it as an error rather than silently ignoring it.               |
| AM-D | `SPEC_EXTENSION` | amici          | Automatic conservation law reduction via L0-matrix stoichiometric null-space analysis (lines 2371–2419). Default-enabled; introduces `tcl_*` moiety parameters not in the original SBML model. |
| ST-C | `SPEC_SILENT`    | SBMLToolbox    | Kinetic law divided by compartment for concentration-canonical species (post-hoc in ODE output). Same effect as copasi (CP-F) but applied selectively per species rather than universally. |
| ST-G | `SPEC_SILENT`    | SBMLToolbox    | fast=true throws MATLAB error. L3v2 §4.11 defines `fast` as removed (undefined behavior); throwing is stricter than ignoring but not explicitly forbidden.                                  |
| ST-H | `SPEC_CONFLICT`  | SBMLToolbox    | Conversion factors throw MATLAB error. Spec §4.11.5 requires conversion factor support.                                                                                                      |
| VC-D | `SPEC_SILENT`    | vcell          | delay() csymbol in kinetic laws: warning logged, expression replaced with 0.0. Model continues with an incorrect kinetic law. Spec §3.4.6 defines delay as valid MathML.                    |
| VC-E | `SPEC_SILENT`    | vcell          | rateOf() csymbol not explicitly handled; no code in SBMLImporter.java. Likely throws at ODE generation. Spec §3.4.8 defines rateOf as valid (L3v2 package).                                 |

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

### amici

Events parsed by `_process_events()` (line 1762 in `importers/sbml/__init__.py`). **Non-zero delays not supported**: delay=0.0 accepted silently; non-zero delay raises `SBMLException("Event delays are currently not supported!")` (lines 828–840, AM-A). **Non-persistent events not supported**: `persistent=false` raises `SBMLException` (lines 868–872, AM-B). Numeric priority stored; non-numeric priority raises `SBMLException`. `useValuesFromTriggerTime` stored. EventAssignment direct concentration or amount assignment per HOSU flag (native; no multiply-by-compartment). Events handled by SUNDIALS root-finding in the generated C++ layer. `rateOf()` natively supported via `_process_sbml_rate_of()` and `_rateof_to_dummy()`.

### SBMLToolbox

Events processed in `WriteEventHandlerFunction.m` (lines 157–167): trigger converted to MATLAB zero-crossing condition. **`persistent=true` OR `initialValue=true` throws** "Cannot deal with persistent trigger" (WriteODEFunction.m line 95–97, ST-J); the same check covers both flags. **Delay throws** "Cannot deal with delayed events" (WriteODEFunction.m line 88–89, ST-D). Priority throws "Cannot deal with priority". `useValuesFromTriggerTime` not implemented — values always evaluated at execution time. EventAssignment written as direct variable assignment in `WriteEventAssignmentFunction.m`; no HOSU amount/concentration conversion applied.

### sbscl

Full event support in `SBMLinterpreter.java` and `EquationSystem.java`. `persistent=false` events aborted when trigger returns false during delay period (lines 221–227 of `SBMLinterpreter.java`). Delay → `delayedEvents` (`List<Integer>`) with execution time checked iteratively each step (lines 229–241 of `SBMLinterpreter.java`). Priority sorting applied at execution (lines 525–544 of `SBMLinterpreter.java`). `useValuesFromTriggerTime=true` → values captured at trigger; `false` → recomputed at execution. EventAssignment to a compartment triggers species amount rebalance (lines 586–589). `initialValue` extracted and stored.

### morpheus

Events converted to MorpheusML `<Event>` elements in `sbml_converter.cpp`. Trigger → `<Condition>` element; `initialValue` → `history` attribute (lines 1089–1098). `persistent` → `persistent` attribute **only when the event also has a delay** (line 1115–1116 is inside the `if (e->isSetDelay())` block); events without delays silently drop `persistent`. Delay + `useValuesFromTriggerTime` → `compute-time` attribute (lines 1100–1116). **Priority dropped with warning** "Dropped unsupported priority element on SBML Event \"%1\"" (lines 1118–1121, MO-D). EventAssignment to amount-unit species: value multiplied by compartment volume (lines 1131–1132). No EventAssignment HOSU branching beyond the amount check.

### vcell

Events handled by `addEvents()` in `SBMLImporter.java` (lines 398–492), creating `BioEvent` objects. Trigger math and delay stored. **`useValuesFromTriggerTime` bug** (VC-K): line 441 calls `event.isSetUseValuesFromTriggerTime()` instead of `getUseValuesFromTriggerTime()`, so `useValuesFromTriggerTime=false` is misread as `true`; also, `useValuesFromTriggerTime` is only read inside the `if (event.isSetDelay())` block — events without delays ignore it entirely. **`initialValue` not extracted** — always treated as false (VC-J). **`persistent` not extracted** — always treated as true (VC-G). **Priority not implemented** — silently dropped (VC-H). EventAssignment stored as direct variable→formula pairs with no HOSU amount/concentration conversion (VC-I). Events only imported for non-spatial models.

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
| AM-A | `SPEC_CONFLICT`  | amici          | Event delay raises `SBMLException`. Spec §4.12.4 defines delay as a required event feature.                                                                                                                                                    |
| AM-B | `SPEC_CONFLICT`  | amici          | `persistent=false` triggers raise `SBMLException`. Spec §4.12.2 requires support for non-persistent triggers.                                                                                                                                  |
| ST-D | `SPEC_CONFLICT`  | SBMLToolbox    | Event delay throws "Cannot deal with delayed events" (WriteODEFunction.m lines 88–89). Spec §4.12.4 defines delay as a required event feature.                                                                                                |
| ST-J | `SPEC_CONFLICT`  | SBMLToolbox    | `persistent=true` OR `initialValue=true` throws "Cannot deal with persistent trigger" (WriteODEFunction.m lines 95–97). Spec §4.12.2 requires support for both persistent triggers and `initialValue=true` events.                             |
| MO-D | `SPEC_SILENT`    | morpheus       | Event priority dropped with warning "Dropped unsupported priority element on SBML Event \"%1\"" (sbml_converter.cpp lines 1118–1121). Spec §4.12.3 defines priority as a valid event feature; dropping changes execution order for simultaneous events.                                |
| VC-G | `SPEC_CONFLICT`  | vcell          | `persistent` attribute not extracted from SBML; all events treated as persistent. Non-persistent event cancellation never occurs.                                                                                                               |
| VC-H | `SPEC_SILENT`    | vcell          | Event priority silently dropped. Spec §4.12.3 defines priority; simultaneous events execute in arbitrary order.                                                                                                                                 |
| VC-I | `SPEC_SILENT`    | vcell          | EventAssignment to species: no HOSU amount/concentration conversion. Spec §4.12.5 requires quantity type consistency. Assignments treated as direct variable writes.                                                                            |
| VC-J | `SPEC_CONFLICT`  | vcell          | `trigger.initialValue` not extracted; always treated as false (trigger can fire at t=0 even when `initialValue=true`). Spec §4.12.2 defines `initialValue=true` as preventing t=0 firing.                                                     |
| VC-K | `SPEC_CONFLICT`  | vcell          | `useValuesFromTriggerTime` bug: `addEvents()` line 441 calls `event.isSetUseValuesFromTriggerTime()` instead of `getUseValuesFromTriggerTime()`, misreading `false` as `true`. Also only read inside `if (event.isSetDelay())` block; events without delays ignore the attribute. Spec §4.12.5 requires correct handling. |

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
| constant species, initialConc, dyn C | ✓       | amount = initConc × C(0)   | t01117, t01118    | ✓                | D13 fixed      |

---

## Appendix C: Adding a New Library

Use the `sbml-add-library` skill (`/sbml-add-library`). It handles source discovery,
Explore-subagent delegation, content writing, and git commit.

Manual steps: add a `### LibraryName` subsection inside each element section (before
`### Divergences`), add a column to the **Feature Matrix**, add a **Library Architectures**
paragraph, and add library-specific divergence IDs (e.g. `XX-A`) to Divergences tables.
