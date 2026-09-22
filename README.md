# siril-scripts

Python scripts for [Siril](https://siril.org) 1.4+, run via the `pyscript` command (Scripts menu or console).

## OSC_Narrowband_Preprocess.py

OSC dual-band narrowband preprocessing for a fixed Ha-OIII and/or SII-OIII filter pair. Converts/calibrates lights, extracts the red-channel line (Ha or SII) and OIII from each session, registers + true-drizzles each, stacks, merges every session's OIII into one combined stack, aligns all final channels together, and background-matches them. Can run headless (edit the `USER SETTING` block) or via a tkinter GUI (`USE_GUI = True`).

## OSC_Narrowband_Multisession_Preprocess.py

Same pipeline, generalized to an arbitrary number of sessions instead of a fixed Ha/SII pair - add or remove sessions freely, each labeled with its own red-channel line (Ha, SII, or a custom name). The GUI manages sessions as a dynamic list (add/remove, per-session lights/darks/flats/biases). Every session's OIII is still merged and stacked once across all sessions for a deeper combined master.

Session concept and UI approach adapted from [Naztronomy's OSC preprocessing script](https://github.com/naztronaut/siril-scripts), reimplemented in plain tkinter (no extra dependencies) instead of PyQt6.

## Usage

Place a script in Siril's scripts folder (or point Siril's Scripts menu at this folder), open the project's working directory in Siril, then run it from Scripts → (script name), or via `pyscript "path\to\script.py"` on the console.

See each script's header comment for the expected input folder layout, output naming, and configuration options.
