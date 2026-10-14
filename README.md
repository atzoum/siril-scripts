# siril-scripts

Python scripts for [Siril](https://siril.org) 1.4+, run via the `pyscript` command (Scripts menu or console).

## OSC_Narrowband_Multisession_Preprocess.py

OSC dual-band narrowband preprocessing for any number of sessions, each tagged with the filter it was shot with (Ha-OIII or SII-OIII). Converts/calibrates each session's lights with its own masters, extracts the red-channel line (Ha or SII) and OIII, then registers + true-drizzles + stacks each channel once across all its sessions: sessions sharing a filter merge into one red-channel stack, and every session's OIII merges into one deeper combined master. All final channels are aligned together on one pixel grid and (optionally) background-matched.

Sessions can also be tagged RGB (no narrowband filter, e.g. for star colour). These are debayered and stacked in full colour instead of channel-split, and the result is aligned on the same pixel grid as the narrowband stacks so it superimposes directly.

Can run via a tkinter GUI (`USE_GUI = True`) or headless (edit the `USER SETTING` block). The GUI manages sessions as a dynamic list: add/remove sessions, and pick each session's lights/darks/flats/biases as files from any folder with "Add Lights/Darks/Flats/Biases" buttons (listed by name and folder, removable individually).

Session concept and UI approach adapted from [Naztronomy's OSC preprocessing script](https://github.com/naztronaut/siril-scripts), reimplemented in plain tkinter (no extra dependencies) instead of PyQt6.

## Usage

Place the script in Siril's scripts folder (or point Siril's Scripts menu at this folder), open the project's working directory in Siril, then run it from Scripts → (script name), or via `pyscript "path\to\script.py"` on the console.

See the script's header comment for the expected input folder layout (headless mode), output naming, and configuration options.
