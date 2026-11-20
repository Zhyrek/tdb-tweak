# tdb-tweak

Hand-tune the parameters of a CALPHAD TDB file (especially ESPEI's `VV0000…` symbols) against a phase diagram that
pycalphad maps again after each change, with literature images and ESPEI ZPF data drawn over it.

## Install

Needs Python 3.11 or newer (3.12 tested).

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
  * **Shared VV symbols.** A VV symbol used in several places gets one row per use, and editing any of them edits
    all of them. A parameter with two VV terms has two rows.
  * **Sorting.** By VV number; by phase → term (G, L0, L1, …, TC, BMAGN) → a, b, c…; or by phase → constituents →
    term.
  * **Elements…** picks the subsystem to edit, for example Fe-Ni-Cr of a Fe-Ni-Co-Cr database. Parameters with any
    other element are left out.
  * **Lower-order parameters** are unaries and binaries in a ternary set. They are red, because changing one also
    changes every subsystem that contains it. A toggle hides them.
  * **Non-VV coefficients** shows the plain numbers in PARAMETER expressions so they can be edited too.
* **Diagram (centre).** The diagram types are binary T–x, ternary isothermal section and isopleth (constant x(C)).
  * **Redraw** (F5 or Ctrl+R) maps again. The mapping runs in the background, lines appear as they are traced, and
    Esc cancels. "Redraw on commit" maps again after every Enter in the table.
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
    * Calibrations are saved next to the image as `*.tdbtweak.json` and in the project.
    * Checking an image switches the diagram to its type, elements and units. **Fit ranges to image** maps only the
      window the image shows, which is faster.
  * **ZPF data.** Load ESPEI datasets as files or a folder. Points are coloured by phase, and tie-lines are dashed.
    * For ternaries, only tie-lines within ±5 K of the section are shown; the tolerance is adjustable.
    * The list of temperatures found in the data maps that section on double-click. Clicking a data point shows its
      reference and enables "Map section at picked point's T".
* **Files.**
  * **Save TDB** writes only the numbers that changed; the rest of the file stays byte-identical.
  * **Projects** (`*.tdbtweak`) store the TDB path, subsystem, diagram settings, image calibrations, ZPF files and
    any unsaved edits.

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

## Caveats

* pycalphad's mapping can be very sensitive to tiny changes. In Al-Cr-Ni at 1273 K, moving one ternary liquid
  coefficient by 1e-6 J/mol changes the number of lines (14 ↔ 16) and drops a B2+B2+L1₂ invariant. If a feature
  appears or vanishes after a small edit, nudge the value or the step size before trusting it.
* The engine relies on pycalphad 0.12 internals (`pycalphad>=0.12,<0.13` is pinned).

## Tests

`.venv/bin/python -m pytest` covers the TDB model, calibration maths, units, the ZPF loader, and the in-place vs
reload maps (both solvers).
