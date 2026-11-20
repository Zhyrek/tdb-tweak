# tdb-tweak

Hand-tune the parameters of a CALPHAD TDB file (especially ESPEI's `VV0000…` symbols) against a phase diagram that
pycalphad maps again after each change, with literature images and ESPEI ZPF data drawn over it.

## Install

Needs Python 3.11 or newer (3.12 tested). Download or clone the repository, then double-click the launcher for your
system:

| System | Launcher |
|---|---|
| Windows | `tdb-tweak.bat` |
| macOS | `tdb-tweak.command` |
| Linux | `tdb-tweak.sh` (`./tdb-tweak.sh --desktop` adds a menu entry) |

The first run creates a private environment (`.venv`) next to the launcher and installs everything, which takes a
few minutes and needs internet. Later runs start the program directly. `--reinstall` repeats the installation. On
Windows the program runs without a console window.

By hand:

```bash
git clone https://github.com/Zhyrek/tdb-tweak.git
cd tdb-tweak
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"         # Windows: .venv\Scripts\pip install -e ".[test]"
.venv/bin/tdb-tweak examples/Cu-Mg/Cu-Mg-mcmc.tdb    # or: .venv/bin/python -m tdb_tweak
```

Optional column-generation solver (needs a C++ compiler):
`.venv/bin/pip install -e ".[cg]"`, which installs [pycalphad-cg](https://github.com/Zhyrek/pycalphad-cg) from GitHub.

Run the tests with `.venv/bin/python -m pytest`.

## Window

* **Parameters (left).** Three columns: symbol, meaning, value.
  * **Meaning.** Phase, parameter and constituents, then the coefficient letter of
    `a + b·T + c·T·ln T + d·T² + e·T³ + f·T⁻¹`. For example, `LIQUID, L1(CU,MG), b` or `CUMG2, G(CU:MG), a`.
    Parameters with several temperature ranges show the range.
  * **Editing.** Double-click a value or start typing, then press Enter. Ctrl+Z / Ctrl+Shift+Z undo and redo.
    Changed values are bold.
  * **Keyboard tweaking.** ↑/↓ moves between parameters. Alt+↑/↓ changes the selected value by one step: 100 J/mol
    for a, 0.1 J/mol/K for b, and the equivalent for c–f. Alt+Shift+↑/↓ changes it by ten steps. Consecutive nudges of
    one value are a single undo step. The diagram redraws about 0.35 s after the last key, using a fast (warm-start)
    map.
  * **Checkpoint and ghost.** Edit ▸ Save Checkpoint (Ctrl+K) records the current parameters. The diagram of the
    checkpoint is drawn faintly in grey behind the current one, so you can see what your changes moved. When a file
    is opened, the checkpoint is its own values. Ctrl+Shift+K reverts to the checkpoint and Ctrl+G shows or hides
    the ghost. With ZPF data loaded, the status line also shows the checkpoint's ZPF score.
  * **Shared VV symbols.** A VV symbol used in several places gets one row per use, and editing any of them edits
    all of them. A parameter with two VV terms has two rows.
  * **Sorting.** By VV number; by phase → term (G, L0, L1, …, TC, BMAGN) → a, b, c…; or by phase → constituents →
    term.
  * **Elements…** picks the subsystem to edit, for example Fe-Ni-Cr of a Fe-Ni-Co-Cr database. Parameters with any
    other element are left out.
  * **Lower-order parameters** are unaries and binaries in a ternary set. They are red, because changing one also
    changes every subsystem that contains it. A toggle hides them.
  * **Non-VV coefficients** shows the plain numbers in PARAMETER expressions so they can be edited too.
  * **Locks.** Right-click rows ▸ Lock (or Ctrl+L) to stop a value from being edited, nudged or varied by the
    optimizer; locked rows show 🔒 and are grey. "Lock every <phase> parameter" locks a whole phase. Locks are shared
    with the optimizer's phase tick boxes and saved in the project.
* **Diagram (centre).** The diagram types are binary T–x, ternary isothermal section and isopleth (constant x(C)).
  * **Redraw** (F5 or Ctrl+R) maps again. The mapping runs in the background, lines appear as they are traced, and
    Esc cancels. "Redraw on commit" maps again after every Enter in the table.
  * **Fast redraws.** Redraws after nudges and commits start from the previous map's starting points instead of
    searching the axis edges again: about 40 % faster on Cu-Mg and 55 % on the Al-Cr-Ni section. The status line marks
    them "(fast)". A brand-new phase region that touches none of the old starting points could be missed, so F5
    always maps in full.
  * **Right-click** on the diagram:
    * **Equilibrium here:** the phases at that point with the current parameters, with their fractions and
      compositions.
    * **G–x curve:** binaries. A window with every phase's Gibbs energy across composition at that temperature,
      relative to the line between the pure elements so the differences are visible. Common tangents of the
      equilibrium are dashed, and line compounds are points.
    * **Step plot:** phase fractions vs temperature at that composition.
    * **Export figure…:** PNG, PDF or SVG. The G–x and step windows have the same entry in their own right-click
      menu.
    The G–x and step windows follow your edits: they recompute 0.3 s after a change.
  * **Diagram ▸ Check for Artifacts** scans the current system for typical CALPHAD breakages:
    * a solid stable at 4000–6000 K;
    * a liquid miscibility gap at those temperatures;
    * a phase that on its own has a miscibility gap at 3000 or 5000 K (an inverted gap);
    * the liquid stable at 300 K, reported as a note.
    The scan also runs automatically after every optimization.
  * **Global-min pdens** is the sampling density of mapping's global-minimum check, which takes most of the mapping
    time. Lowering it is faster but can miss phases.
  * **Solver** switches between pycalphad's solver and column generation.
* **Views bar (under the diagram controls).** A row of buttons, one per saved diagram, for the diagrams you keep
  coming back to (e.g. a zoomed binary region, the 1273 K and 1473 K sections, an isopleth), as opposed to a
  temperature you just want a quick look at.
  * **+ Save view** stores the current diagram as a button: type, elements, map ranges, section temperature or fixed
    x(C), units, solver, global-min pdens and tie-lines, plus the zoomed region if you zoomed with the toolbar. The
    suggested name can be edited.
  * **Switching:** click a button or press Ctrl+1…9. The active view is highlighted until you change the diagram
    settings.
  * **Right-click** a button to update it to the current diagram and zoom, rename it, move it, or delete it.
  * **Map all views** maps every saved view with the current values, one after another in the background, the active
    one first. Views not mapped with the current values are italic.
  * **Result cache:** every finished map is kept, keyed by its settings and the exact parameter values. Switching to
    an up-to-date view, or pressing F5 with nothing changed, shows it instantly. After an edit, the view's previous
    map stays on screen until the new one arrives.
  * The phase selection (Phases…) is shared by all views.
  * Views are saved in the project file.
* **Overlays (right).**
  * **Literature images.** Add an image, then calibrate it: diagram type, elements, units (K/°C/°F, at.%/wt.%/
    fractions) and the axis bounds to show.
    * Binary and isopleth images: click two points on each axis with known values. They do not have to be the
      corners or round numbers.
    * Any image: click N points with both values known. 3 points give an affine fit; 4 or more with "perspective"
      give a homography.
    * Partial triangles and composition windows work.
    * A magenta grid drawn back onto the image shows whether the calibration fits.
    * **Frame detection:** for binary and isopleth images, the plot frame is found automatically, even when the
      image is slightly rotated or sheared. The four axis points are placed on its corners, with the axis bounds as
      their values, so you only check them. Markers can be dragged, and "Detect frame" runs the detection again.
    * Calibrations are saved next to the image as `*.tdbtweak.json` and in the project.
    * Checking an image switches the diagram to its type, elements and units. **Fit ranges to image** maps only the
      window the image shows, which is faster.
  * **ZPF data.** Load ESPEI datasets as files or a folder. Points are coloured by phase, and tie-lines are dashed.
    * For ternaries, only tie-lines within ±5 K of the section are shown; the tolerance is adjustable.
    * The list of temperatures found in the data maps that section on double-click. Clicking a data point shows its
      reference and enables "Map section at picked point's T".
    * **ZPF score:** after each map, the status line shows how far the data are from equilibrium with the current
      parameters, e.g. "ZPF 699 J/mol (530 pts)". This follows ESPEI's approach:
      * the target hyperplane is the mean chemical potential of the single-phase equilibria at a tie-line's measured
        vertices;
      * each vertex's driving force is its phase's distance from that hyperplane;
      * the score is the RMS in J/mol, where 0 is a perfect fit.
      Each dataset in the list shows its own score, and the status-line tooltip lists the worst points.
* **Optimize (binary invariants and congruent points, ternary tie triangles).** Moves a three-phase invariant (eutectic, peritectic, …)
  to a target temperature and/or target compositions of its three phases, or a congruent point (congruent melting
  or a congruent solid-state transformation) to a target temperature and composition, by adjusting the parameters
  you choose.
  * **Congruent points** are found on every binary map and marked ◇. The marker is the exact point, solved from
    equal chemical potentials of both elements in the two phases at one composition; that is accurate to about
    1 mK and takes a few ms. Right-click one and choose "Optimize congruent point…", then drag the point. If one
    phase is a line compound (e.g. CuMg₂), the composition is fixed at its stoichiometry and only T can be dragged
    or targeted.
  * **Targets:** right-click close to an invariant line or a congruent point and choose "Optimize…". The point is
    added to the target list in the Optimize tab; the "On this map" list can add points too. Every target is drawn
    with numbered handles:
    * invariant: drag the dashed line up or down for T, and each end point sideways for that phase's composition;
    * congruent point: drag the point.
    The Want column shows the same values and can be typed into; tick boxes leave single values out.
  * **Ternary tie triangles:** on an isothermal section, right-click inside (or near) a three-phase triangle and
    choose "Optimize tie triangle…", then drag its corners. The target is the composition of each corner; T is the
    section's. At fixed T and P the three corners have no freedom left, so each one is solved directly (pycalphad's
    solver with the overall composition inside the triangle).
  * **Stoichiometry:** each phase's reachable compositions come from its sublattice model:
    * a single composition (line compound): its handle is grey and cannot move, and its composition is not a target;
      on a binary invariant, pressing it drags the temperature line instead;
    * a line, e.g. (Fe)(Ni,Cr), where x(Fe) is fixed and Ni and Cr exchange: its handle slides along that line, and
      typed targets are projected onto it;
    * otherwise free.
    This applies to binary invariants (e.g. CuMg₂ in the CuMg₂ + liquid + Laves eutectic), congruent points and tie
    triangles.
  * **Robust starts:** if the map used for the optimisation does not contain the feature, the starting phase states
    are built from the target's compositions: each phase is sampled at T and the lowest-energy state near the
    corner is used. pycalphad's mapping can find different triangles for equal energies compiled in two ways, so
    plain coefficients chosen as parameters are converted at once and the diagram is mapped again, before you pick
    targets.
  * **Several targets at once:** all targets are optimised together. A target you add but don't drag keeps its
    current position as the goal, so it **holds that feature in place**. For example, move the Laves melting point
    while holding the CuMg₂ melting point and the Mg-side eutectic.
  * **Parameters:** "VV rows shown in table" uses every VV symbol listed in the parameter table after its filters;
    "Selected table rows" uses only the selected rows, VV or plain coefficients. Any parameter can be unticked.
  * **Freezing phases:** "Vary parameters of:" has one tick box per phase. Untick a phase, e.g. LIQUID, to freeze
    every parameter of that phase; a VV symbol shared with a frozen phase is frozen too. The freeze is kept when the
    list is refilled. This is the simplest way to stop a fit from changing a phase that affects the whole diagram.
  * **Method:** each iteration solves the invariant directly (pycalphad's solver with the middle-composition phase
    fixed at zero amount and T free, about 0.3 ms). It builds a finite-difference Jacobian and takes a damped
    least-squares (Levenberg–Marquardt) step in limit-scaled parameters. That is the most downhill step for the
    linearised problem and, with more parameters than targets, the smallest one.
  * **Per-iteration limit:** each parameter changes by at most max(100 % of its current value, floor). The floors
    are a 50000 J/mol, b 50 J/mol/K, c 7, d 0.05, e 5e-5, f 5e7, each about 50 kJ/mol at 1000 K. The percentage,
    the floors and per-parameter overrides ("Max Δ / iter") are editable.
  * **Norm term (optional):** adds weight × Σ ((p − centre)/floor)² to the cost, where a target off by one tolerance
    costs 1. The centre is zero (keep parameters small) or the starting values (stay close to them).
    * 0.001: does almost nothing unless a parameter grows to ~100× its floor, i.e. it only stops blow-ups;
    * 1: strong;
    * 100: keeps parameters as small as possible, even at the cost of the targets.
    With the norm on, iterations continue after the targets are met while the total still decreases. In the Cu-Mg
    test, a weight of 0.05 toward zero reached the same Laves target with parameters about half as large.
  * **Stopping:** when every target is within tolerance (1 K, 0.1 at.% by default) or after the maximum number of
    iterations.
  * **Metastable targets:** a step that makes a target metastable (another phase becomes more stable there) is
    rejected and shortened by default. The run stops if every step toward the goal would do that. With the option
    off, such iterations are accepted and flagged in the history.
  * **Undo:** the final iteration is applied as one undo step and the diagram is mapped again. The iteration
    history lets you apply any earlier iteration instead, e.g. "a few steps in that direction", or revert to the
    start. Ctrl+Z works too. The change table shows each parameter's start value, value and Δ/limit; red means
    the parameter hit its limit.
  * **Boundary nodes:** right-click a phase boundary line of a binary ▸ "Add boundary node", then drag the node. The
    target is that boundary passing through the node: at the node's temperature, the phase's boundary composition
    is solved as a two-phase equilibrium and compared with the node's x.
  * **Targets across views:** each target belongs to the diagram it was added on, which is shown in the target
    list. One run can mix targets from binary diagrams and from ternary sections at different temperatures. Run it
    from any diagram; "Map all views" refreshes the others afterwards.
  * **Hold everything else:** one tick box. Every other invariant, congruent point and tie triangle of the targets'
    diagrams is then held where it is, within 5 K / 1 at.%. In the Cu-Mg test, moving the Laves maximum by 26 K moved
    nothing else by more than about 2 K; without the hold, other features moved 130–180 K.
  * **Fit the loaded ZPF data** (tick box): the driving forces of the loaded ZPF datasets of the targets' systems are
    added as residuals, with σ = 1000 J/mol per point. This works together with dragged targets, holds, locks and
    the norm, or alone, as a pure data fit of the chosen parameters. The history shows the ZPF score of each
    iteration.
  * **Reject steps that create artifacts** (on by default): every candidate step is checked with the artifact scan,
    and a step that creates a new one is shortened instead. Example, Cu-Mg from the parameter-generation database:
    * data-only fit without the guard: ZPF 1379 → 176 J/mol, but with high-temperature miscibility gaps in the
      liquid, FCC and Laves phases;
    * with the guard: 1379 → 209 J/mol, no artifacts, and much smaller parameter changes.
  * **Runaway guard:**
    * **total-change cap:** over one run, a parameter moves at most N× its per-iteration limit, 5 by default;
    * **dominance note:** when one parameter carries more than 80 % of a step, the history says so, because the
      other parameters barely affect the targets and that one is being pushed hard.
  * **Side effects:** only the targets are constrained. Other features (e.g. a neighbouring eutectic) can still move,
    so add the ones that matter as held targets or use "Hold everything else", and vary only the parameters of the
    phases taking part.
* **Compare with another TDB** (File menu): draws another file, such as a published assessment or an earlier
  version, as the grey ghost instead of the checkpoint. With ZPF data loaded, its score appears next to yours, e.g.
  "ZPF 700 J/mol (Cu-Mg-generated.tdb 1379)". Both databases stay compiled side by side, so redraws are not slowed.
  File ▸ Stop Comparing goes back to the checkpoint ghost.
* **ESPEI MCMC polish** (Tools menu): a short ESPEI MCMC run from the current values, using your own ESPEI
  installation. You choose its Python, e.g. an environment with the fast `backend: cg`. Give an ESPEI input YAML as
  a template so its tags, prior and data weights are kept, or the phase models and datasets folder. Only unlocked VV
  parameters are fitted, through ESPEI's `symbols`. The best sample is applied as one undo step. Example: Cu-Mg, 21
  parameters, 5 iterations with `backend: cg` took 8 s and raised ln p from −4886 to −4814.
* **Files.**
  * **Save TDB** writes only the numbers that changed; the rest of the file stays byte-identical.
  * **Projects** (`*.tdbtweak`) store the TDB path, subsystem, diagram settings, image calibrations, ZPF files and
    any unsaved edits.
  * **Last session:** every minute and on quitting, the session is saved and reopened at the next start (start with
    `--fresh` to skip that). Unsaved edits are kept by the periodic save, as crash recovery, but not when you quit
    with "Discard". They are restored only if the TDB file is unchanged.
  * **Save as Next Version** (Ctrl+Alt+V) writes `name_v2.tdb`, then `_v3`, and so on. **Show Changes…** (Ctrl+D)
    is a line diff of your edits against the file as opened or last saved.

## Why redraws are fast

Each editable number is a pycalphad *parameter*: a symbol whose value is supplied at run time. The models and
compiled energy functions are built once for each database structure, element set and phase set. An edit only writes
the new value into `PhaseRecordFactory.param_values`, so nothing is re-parsed or recompiled.

* **VV symbols** are always parameters.
* **A plain coefficient** becomes a parameter the first time it is edited. That one change recompiles once; after
  that it is as fast as a VV edit.

pycalphad's mapping does not pass parameter values to the `Workspace` it uses for starting points. On 0.12 that
fails with "Parameter symbol mismatch"; on 0.11 it silently used the values stored in the database. `engine.py`
patches this. `tests/test_engine.py` checks that an in-place edit gives exactly the map of a full reload, with both
solvers.

Measured on an M-series Mac, pycalphad 0.12.0:

| System | Map after an edit |
|---|---|
| Cu-Mg binary, 500–1500 K, pycalphad solver | ~0.75 s |
| Cu-Mg binary, column generation | ~0.5 s |
| Al-Cr-Ni 1273 K section, 2 % steps | ~3 s |
| Al-Cr-Ni isopleth, x(Cr) = 0.1 | ~14 s (pycalphad's isopleth strategy) |
| Fast redraw (warm start) after a nudge: Cu-Mg / Al-Cr-Ni section | ~0.5 s / ~1.4 s |
| "Map all views", 4 Cu-Mg views / 3 Al-Cr-Ni sections (process pool, warm) | 1.2 s / 3.2 s (sequential: 2.9 s / 8.4 s) |

"Map all views" uses a pool of worker processes. It is started in the background once two views are saved, and its
first run includes process start-up.

## Caveats

* pycalphad's mapping can be very sensitive to tiny changes. In Al-Cr-Ni at 1273 K, moving one ternary liquid
  coefficient by 1e-6 J/mol changes the number of lines (14 ↔ 16) and drops a B2+B2+L1₂ invariant. If a feature
  appears or vanishes after a small edit, nudge the value or the step size before trusting it.
* The engine relies on pycalphad 0.12 internals (`pycalphad>=0.12,<0.13` is pinned).

## Later ideas

* **Map cross-check:** re-check a finished map with point equilibria (or the column-generation solver) on a coarse
  grid and mark lines where they disagree. This would matter if pycalphad's mapping fragility turns out to be a
  frequent problem.
* **Parameter ties:** keep e.g. L1 = 0 or a fixed b/a ratio during optimization (locks cover the common case).
* **Literature points by hand + automatic classification:** you mark points on an image, and a model classifies
  them (liquidus, solvus, invariant, which phases) and proposes corrections. This would be an alternative to
  automatic digitizing, which is fragile on real figures. They could then become targets or ESPEI ZPF datasets.

## Tests

`.venv/bin/python -m pytest` covers the TDB model, calibration maths, units, the ZPF loader, and the in-place vs
reload maps (both solvers).
