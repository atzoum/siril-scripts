# OSC_Narrowband_Multisession_Preprocess.py
#
# Siril 1.4+
#
# OSC dual-band narrowband preprocessing with an arbitrary number of
# sessions (not just a fixed Ha-OIII / SII-OIII pair - add as many
# sessions as you have). Each session is tagged with the dual-band
# filter it was shot with, "Ha-OIII" or "SII-OIII" - no custom/free-
# text labels. Sessions sharing the same filter have their red
# channel (Ha for Ha-OIII, SII for SII-OIII) merged into ONE combined
# stack (e.g. two Ha-OIII sessions from different nights -> one
# deeper Ha master), exactly like every session's OIII always merges
# into ONE combined stack regardless of filter. All resulting
# channels (one per unique filter's red channel + 1 combined OIII)
# are then aligned together on a shared pixel grid.
#
# Session concept and UI approach adapted from Naztronomy's OSC
# preprocessing script (https://github.com/naztronaut/siril-scripts),
# reimplemented in plain tkinter (no extra dependencies) instead of
# PyQt6.
#
# Siril's seqextract_HaOIII always splits Bayer CFA data into a
# "red channel" sequence and a "green+blue channel" sequence and
# always names them Ha_*/OIII_* internally, regardless of which real
# narrowband line the red channel represents - a session's filter
# below only controls how ITS red channel is named (Ha or SII) and
# its default folder names; the extraction itself is identical either way.
#
# Can be run two ways:
#
#   USE_GUI = True   -> shows a dialog to manage sessions and pick
#                        options, for interactive use from Siril's
#                        GUI. Sessions can be added/removed freely.
#   USE_GUI = False  -> uses the SESSIONS list and USER SETTING
#                        values below as-is, for headless use.
#
# INPUT FOLDERS (headless mode; per session, defaults under Siril's
# working dir unless a session's lights/darks/flats/biases override
# is set):
#
#   lights_haoiii/ or lights_siioiii/    REQUIRED (per session; skip
#                                        a session with an empty/
#                                        missing lights folder)
#   darks_haoiii/ or darks_siioiii/      optional
#   flats_haoiii/ or flats_siioiii/      optional
#   biases_haoiii/ or biases_siioiii/    optional
#
# (folder suffix = the session's filter, lowercased with the hyphen
# removed, e.g. "Ha-OIII" -> haoiii)
#
# OUTPUT:
#
# Saved directly into Siril's current working folder. Each unique
# filter's combined red channel gets its own name built from ALL its
# member sessions' light frames put together (common filename
# prefix, kept whole through a shared date even though per-frame
# time differs, plus "<n>x<exposure>s" or "<n>f"); the combined
# OIII's name is built the same way from EVERY session's light
# frames put together:
#
#   <lights prefix>_Ha_x<scale>.fit    (if any Ha-OIII session given)
#   <lights prefix>_SII_x<scale>.fit   (if any SII-OIII session given)
#   <lights prefix>_OIII_x<scale>.fit  (from ALL sessions' OIII)
#
#
# NORMALIZATION
# -------------
#
# If MATCH_BACKGROUNDS is True (the default), every non-reference
# channel gets its background level (median) shifted to match the
# reference channel's median, via PixelMath. The reference is "Ha"
# if any Ha session was given, otherwise "SII". Contrast/noise (MAD)
# is left untouched on every channel - this only gives every channel
# the same black point for a clean composite, without amplifying a
# fainter channel's noise the way matching contrast/scale would. It
# is not a scientific calibration. The reference channel itself is
# always saved as-is, un-normalized. If MATCH_BACKGROUNDS is False,
# every channel (including OIII) is saved exactly as stacked, with
# no background matching at all.
#
#
# SCALE behaviour
# ---------------
#
# SCALE = 1
#
#   Each session's red channel:
#       half-res extraction -> TRUE 2x drizzle -> native full
#       sensor resolution
#
#   OIII (combined across sessions):
#       native full-resolution extraction -> normal registration
#
# SCALE > 1
#
#   Each session's red channel:
#       half-res extraction -> TRUE 2x drizzle -> native full
#       sensor resolution -> Lanczos upscale by SCALE
#
#   OIII:
#       native full-resolution extraction -> TRUE drizzle directly
#       by SCALE
#
# Siril drizzle supports scales up to 3. Note: a session's red
# channel is ALWAYS true-drizzled 2x regardless of SCALE - SCALE
# only adds further scaling on top of that.
#
# All registration -> stack steps use -framing=min, cropping each
# sequence to the area common to every frame. This avoids the bright
# edge artefacts that show up where dithered subs only partially
# overlap (fewer contributing frames there skews normalized/rejected
# stacking at the border). The final cross-channel alignment step
# assumes every session frames the same target/field - it registers
# all final channel masters together and will simply fail to match
# stars if that assumption doesn't hold.
#

import os
import re
import shutil
from collections import Counter
from pathlib import Path

import sirilpy as s
from sirilpy import SirilConnectionError


# ============================================================
# USER SETTING (used directly when USE_GUI = False)
# ============================================================

USE_GUI = True

# Drizzle scale factor (1.0 - 3.0). Note: every session's red channel
# is ALWAYS true-drizzled 2x regardless of this setting - see SCALE
# behaviour in the header comment above.
SCALE = 1.0

# Drizzle pixel fraction ("pixfrac", 0 < value <= 1). Smaller values
# sharpen the drizzled result but need more, better-dithered frames
# to avoid gaps; 0.8 is a reasonable general default.
PIXFRAC = 0.8

# If True, the process/masters folders are wiped before this run
# starts AND removed again after it finishes successfully, leaving
# only the final results. Set False to keep those intermediate files
# around (e.g. to resume after an earlier step without redoing
# conversion, calibration or registration).
CLEANUP_PREVIOUS = True

# Per-frame weighting applied when stacking registered sequences
# (not applied to bias/dark/flat master stacks). One of: None,
# "noise", "wfwhm", "nbstars", "nbstack". None disables weighting
# (equal weight per frame).
STACK_WEIGHT = None

# If True, every non-reference channel's background (median) is
# shifted to match the reference channel's - see NORMALIZATION in
# the header comment above. If False, every channel (including OIII)
# is saved exactly as stacked, with no background matching at all.
MATCH_BACKGROUNDS = True

# One entry per session. "filter" must be "Ha-OIII" or "SII-OIII" -
# the dual-band filter this session was shot with. It sets the
# session's default folder names (see INPUT FOLDERS above) and, via
# its red channel ("Ha" for a Ha-OIII filter, "SII" for SII-OIII),
# the session's output filename. Add or remove entries freely; a
# session with no light frames is skipped. Two sessions with the
# same filter (e.g. two Ha-OIII sessions from different nights) have
# their red channels merged into one combined stack, same as OIII
# always does across every session regardless of filter.
SESSIONS = [
    {"filter": "Ha-OIII", "lights": None, "darks": None, "flats": None, "biases": None},
    {"filter": "SII-OIII", "lights": None, "darks": None, "flats": None, "biases": None},
]

# ============================================================


# Fixed drizzle setting
KERNEL = "square"

STACK_WEIGHT_CHOICES = ("noise", "wfwhm", "nbstars", "nbstack")

# Human-readable labels for the GUI dropdown, per Siril's own -weight=
# documentation (siril "stack" command help).
STACK_WEIGHT_LABELS = {
    None: "None (equal weight per frame)",
    "noise": "Noise (favor lower-noise frames)",
    "wfwhm": "Star sharpness / wFWHM (favor sharper frames)",
    "nbstars": "Star count (favor frames with more detected stars)",
    "nbstack": "Sub-stack count (for live-stacked/pre-stacked inputs)",
}
WEIGHT_LABEL_TO_VALUE = {label: value for value, label in STACK_WEIGHT_LABELS.items()}

# Matches a per-frame exposure time token like "_300s" or "_120.0s"
# (e.g. from "..._300s60_..." or "..._120.0s_Bin1_..."). Requires a
# delimiter or start-of-string right before the digits so it doesn't
# match unrelated numbers (e.g. "Bin1", "mk127").
EXPOSURE_RE = re.compile(r"(?:^|[_\-])(\d+(?:\.\d+)?)s", re.IGNORECASE)

# A session's "filter" is the actual dual-band filter it was shot
# with (shown in the UI and used for default folder names); its red
# channel is the shorter name used for output filenames/grouping.
FILTER_TYPES = ("Ha-OIII", "SII-OIII")
FILTER_TO_CHANNEL = {"Ha-OIII": "Ha", "SII-OIII": "SII"}


def resolve_dir(value, root, name):
    return Path(value) if value else (root / name)


def slugify(filter_type):
    """Folder-name-safe version of a filter type, e.g. "Ha-OIII" -> "haoiii"."""
    slug = re.sub(r"[^A-Za-z0-9]+", "", filter_type or "").lower()
    return slug or "session"


def sanitize_token(text):
    """
    Collapse whitespace to underscores. Siril's cmd() joins arguments
    with plain spaces (no quoting), so a space in a filename/sequence
    name would break commands like "save". Returns "" if nothing is
    left once sanitized.
    """
    text = re.sub(r"\s+", "_", (text or "").strip())
    text = re.sub(r"_+", "_", text)
    return text


def default_session_dirs(root, filter_type, overrides):
    """
    Resolve a session's 4 folder settings against their
    "<kind>_<slug(filter_type)>" defaults under root, e.g.
    filter_type="Ha-OIII" -> lights_haoiii, darks_haoiii, ...
    """
    slug = slugify(filter_type)
    return (
        resolve_dir(overrides.get("lights"), root, f"lights_{slug}"),
        resolve_dir(overrides.get("darks"), root, f"darks_{slug}"),
        resolve_dir(overrides.get("flats"), root, f"flats_{slug}"),
        resolve_dir(overrides.get("biases"), root, f"biases_{slug}"),
    )


def list_light_files(directory):
    if directory is None or not directory.is_dir():
        return []

    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


def has_files(directory):
    return bool(list_light_files(directory))


def parse_exposure_seconds(stem):
    """
    Find a per-frame exposure time (in seconds) in a single light
    frame's filename, e.g. "..._300s60_..." -> 300.0. Returns None
    if no such token is found.
    """
    match = EXPOSURE_RE.search(stem)
    return float(match.group(1)) if match else None


def build_exposure_suffix(stems, count):
    """
    Build a "<n>x<exposure>s" suffix from every light frame's parsed
    exposure time, e.g. all 17 frames at 300s -> "17x300s"; a mix of
    2 frames at 20s and 10 at 10s -> "2x20s+10x10s" (longest exposure
    first). Falls back to just "<n>f" if any frame's exposure can't
    be parsed.
    """
    exposures = []

    for stem in stems:
        exposure = parse_exposure_seconds(stem)
        if exposure is None:
            return f"{count}f"
        exposures.append(exposure)

    counts = Counter(exposures)

    parts = [
        f"{n}x{exposure:g}s"
        for exposure, n in sorted(counts.items(), reverse=True)
    ]

    return "+".join(parts)


def derive_base_name_from_files(files):
    """
    Build an output base name from a set of light frames: the common
    filename prefix (trimmed back to a whole token on the last "_"
    or "-", whichever comes later, e.g.
    "Unknown_300s60_Duo-Band_20260919-220535997_13C" ->
    "Unknown_300s60_Duo-Band_20260919" - this keeps a shared date
    even though per-frame time-of-day differs) plus "<n>x<exposure>s"
    (e.g. "17x300s") if every frame has a parseable exposure time,
    otherwise just "<n>f".
    """
    count = len(files)

    stems = [p.stem for p in files]
    common = os.path.commonprefix(stems) if stems else ""

    cut = max(common.rfind("_"), common.rfind("-"))
    if cut > 0:
        common = common[:cut]

    common = sanitize_token(common.strip("_- "))

    if not common:
        common = "narrowband"

    suffix = build_exposure_suffix(stems, count)

    return f"{common}_{suffix}"


def prompt_settings(root):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    win = tk.Tk()
    win.title("OSC Narrowband Preprocess (multi-session)")

    style = ttk.Style()
    style.configure("Header.TLabel", font=("TkDefaultFont", 9, "bold"))
    style.configure("Note.TLabel", font=("TkDefaultFont", 8), foreground="#666666")

    outer = ttk.Frame(win, padding=10)
    outer.grid(sticky="nsew")

    # Fixed wraplength wide enough to span the window's actual content
    # width (set by the wider entry/combobox rows below). A dynamic,
    # resize-driven wraplength was tried and removed: since this window
    # auto-sizes to fit its content, changing a label's wraplength on
    # <Configure> changes its requested size, which resizes the window,
    # which fires another <Configure> - an infinite resize loop.
    WRAP = 560

    intro_label = ttk.Label(
        outer,
        text="Combines one or more Ha-OIII / SII-OIII imaging sessions "
             "into separate Ha, SII and OIII stacks. Your camera's "
             "sensor splits colors in a checkerboard pattern, so the "
             "red channel starts at half resolution - this script "
             "drizzles it back up to full size.",
        wraplength=WRAP, justify="left"
    )
    intro_label.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 10))

    scale_var = tk.StringVar(value=f"{SCALE:g}")
    pixfrac_var = tk.StringVar(value=f"{PIXFRAC:g}")
    cleanup_var = tk.BooleanVar(value=CLEANUP_PREVIOUS)
    match_backgrounds_var = tk.BooleanVar(value=MATCH_BACKGROUNDS)
    weight_var = tk.StringVar(value=STACK_WEIGHT_LABELS[STACK_WEIGHT])

    # --------------------------------------------------------
    # Session state: one dict of Tk variables per session. The
    # editor panel's widgets get their textvariable/variable REBOUND
    # to the selected session's vars on each selection change, and
    # every widget command looks the current session up dynamically
    # (via cur()) rather than closing over a specific var, so it
    # always acts on whichever session is selected at click-time.
    # --------------------------------------------------------

    sessions = []
    current = {"index": 0}

    def cur():
        return sessions[current["index"]]

    def make_session_vars(filter_type, lights_d, darks_d, flats_d, biases_d):
        return {
            "filter_var": tk.StringVar(value=filter_type),
            "lights_var": tk.StringVar(value=str(lights_d)),
            "darks_var": tk.StringVar(value=str(darks_d)),
            "darks_enabled": tk.BooleanVar(value=has_files(darks_d)),
            "flats_var": tk.StringVar(value=str(flats_d)),
            "flats_enabled": tk.BooleanVar(value=has_files(flats_d)),
            "biases_var": tk.StringVar(value=str(biases_d)),
            "biases_enabled": tk.BooleanVar(value=has_files(biases_d)),
        }

    for entry in SESSIONS:
        filter_type = entry.get("filter") or FILTER_TYPES[0]
        dirs = default_session_dirs(root, filter_type, entry)
        sessions.append(make_session_vars(filter_type, *dirs))

    if not sessions:
        sessions.append(make_session_vars(FILTER_TYPES[0], *default_session_dirs(root, FILTER_TYPES[0], {})))

    # --------------------------------------------------------
    # Left: session list + add/remove
    # --------------------------------------------------------

    left = ttk.Frame(outer)
    left.grid(row=1, column=0, sticky="ns", padx=(0, 10))

    ttk.Label(left, text="Sessions", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")

    listbox = tk.Listbox(left, width=32, height=10, exportselection=False)
    listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(2, 4))

    def describe(index):
        sv = sessions[index]
        filter_type = sv["filter_var"].get() or FILTER_TYPES[0]
        lights_text = sv["lights_var"].get()
        n = len(list_light_files(Path(lights_text))) if lights_text else 0
        return f"{index + 1}: {filter_type} ({n} lights)"

    def refresh_listbox(select_index=None):
        listbox.delete(0, "end")
        for i in range(len(sessions)):
            listbox.insert("end", describe(i))
        target = current["index"] if select_index is None else select_index
        target = max(0, min(target, len(sessions) - 1))
        listbox.selection_clear(0, "end")
        listbox.selection_set(target)
        select_session(target)

    def add_session():
        filter_type = FILTER_TYPES[0]
        sv = make_session_vars(filter_type, *default_session_dirs(root, filter_type, {}))
        sv["filter_var"].trace_add("write", on_label_change)
        sessions.append(sv)
        refresh_listbox(select_index=len(sessions) - 1)

    def remove_session():
        if len(sessions) <= 1:
            messagebox.showerror("Error", "At least one session is required.")
            return
        idx = current["index"]
        sessions.pop(idx)
        refresh_listbox(select_index=min(idx, len(sessions) - 1))

    ttk.Button(left, text="+ Add Session", command=add_session).grid(row=2, column=0, sticky="we")
    ttk.Button(left, text="− Remove Session", command=remove_session).grid(row=2, column=1, sticky="we")

    # --------------------------------------------------------
    # Right: editor panel for the currently selected session
    # --------------------------------------------------------

    editor = ttk.Labelframe(outer, text="Session", padding=8)
    editor.grid(row=1, column=1, sticky="nsew")

    def set_state(widget, enabled):
        widget.state(["!disabled"] if enabled else ["disabled"])

    erow = 0

    ttk.Label(editor, text="Filter type:").grid(row=erow, column=0, sticky="w", pady=(0, 8))
    filter_combo = ttk.Combobox(editor, values=list(FILTER_TYPES), width=17, state="readonly")
    filter_combo.grid(row=erow, column=1, sticky="w", pady=(0, 8))
    erow += 1

    def browse_into(var):
        # Only start from the field's current value if it's a real,
        # existing folder (it's usually still a not-yet-created
        # default like ".../lights_haoiii") - otherwise start from
        # Siril's working directory (its "home").
        existing = var.get()
        start = existing if existing and Path(existing).is_dir() else str(root)
        path = filedialog.askdirectory(initialdir=start)
        if path:
            var.set(path)
            refresh_listbox()

    def folder_row(row_label, var_key, enabled_key=None):
        nonlocal erow
        this_row = erow

        if enabled_key is not None:
            cb = ttk.Checkbutton(
                editor, text=row_label,
                command=lambda: apply_editor_states()
            )
            cb.grid(row=this_row, column=0, sticky="w")
        else:
            cb = None
            ttk.Label(editor, text=f"{row_label}:").grid(row=this_row, column=0, sticky="w")

        entry = ttk.Entry(editor, width=42)
        entry.grid(row=this_row, column=1, sticky="we")

        btn = ttk.Button(editor, text="Browse...", command=lambda: browse_into(cur()[var_key]))
        btn.grid(row=this_row, column=2)

        erow += 1
        return cb, entry, btn

    lights_cb, lights_entry, lights_btn = folder_row("Lights", "lights_var")
    darks_cb, darks_entry, darks_btn = folder_row("Darks", "darks_var", "darks_enabled")
    flats_cb, flats_entry, flats_btn = folder_row("Flats", "flats_var", "flats_enabled")
    biases_cb, biases_entry, biases_btn = folder_row("Biases", "biases_var", "biases_enabled")

    def apply_editor_states():
        sv = cur()
        for cb, entry, btn, enabled_key in (
            (darks_cb, darks_entry, darks_btn, "darks_enabled"),
            (flats_cb, flats_entry, flats_btn, "flats_enabled"),
            (biases_cb, biases_entry, biases_btn, "biases_enabled"),
        ):
            enabled = sv[enabled_key].get()
            set_state(entry, enabled)
            set_state(btn, enabled)

    def select_session(index):
        current["index"] = index
        sv = sessions[index]

        filter_combo.configure(textvariable=sv["filter_var"])
        lights_entry.configure(textvariable=sv["lights_var"])
        darks_entry.configure(textvariable=sv["darks_var"])
        darks_cb.configure(variable=sv["darks_enabled"])
        flats_entry.configure(textvariable=sv["flats_var"])
        flats_cb.configure(variable=sv["flats_enabled"])
        biases_entry.configure(textvariable=sv["biases_var"])
        biases_cb.configure(variable=sv["biases_enabled"])

        apply_editor_states()

    def on_listbox_select(_event):
        sel = listbox.curselection()
        if sel:
            select_session(sel[0])

    listbox.bind("<<ListboxSelect>>", on_listbox_select)

    def on_label_change(*_args):
        idx = listbox.curselection()
        if idx:
            listbox.delete(idx[0])
            listbox.insert(idx[0], describe(idx[0]))
            listbox.selection_set(idx[0])

    refresh_listbox(select_index=0)

    # Re-describe the listbox row when the filter type changes, so it
    # doesn't go stale while editing.
    for sv in sessions:
        sv["filter_var"].trace_add("write", on_label_change)

    # --------------------------------------------------------
    # Shared options
    # --------------------------------------------------------

    opts = ttk.Frame(outer)
    opts.grid(row=2, column=0, columnspan=2, sticky="we", pady=(10, 0))

    ttk.Label(opts, text="Options", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")

    ttk.Label(opts, text="Drizzle scale:").grid(row=1, column=0, sticky="w", pady=(4, 0))
    ttk.Combobox(
        opts, textvariable=scale_var, values=["1", "1.5", "2", "2.5", "3"], width=8
    ).grid(row=1, column=1, sticky="w", pady=(4, 0))

    ttk.Label(
        opts,
        text="The red channel is always 2x drizzled first, then scaled "
             "up further if this is set above 1. Other channels follow "
             "this setting directly.",
        style="Note.TLabel", wraplength=WRAP, justify="left"
    ).grid(row=2, column=0, columnspan=3, sticky="we", pady=(0, 4))

    ttk.Label(opts, text="Drizzle pixel fraction:").grid(row=3, column=0, sticky="w", pady=4)
    ttk.Combobox(
        opts, textvariable=pixfrac_var, values=["0.5", "0.65", "0.8", "0.9", "1.0"], width=8
    ).grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(opts, text="Stack weighting:").grid(row=4, column=0, sticky="w", pady=4)
    ttk.Combobox(
        opts, textvariable=weight_var, values=list(STACK_WEIGHT_LABELS.values()),
        width=48, state="readonly"
    ).grid(row=4, column=1, columnspan=2, sticky="we", pady=4)

    ttk.Checkbutton(
        opts, text="Match backgrounds (median-only normalization)",
        variable=match_backgrounds_var
    ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))

    ttk.Label(
        opts,
        text="Aligns each channel's background brightness to the "
             "reference (Ha if present, else SII) without touching "
             "contrast, so it won't amplify noise. Off leaves every "
             "channel exactly as stacked.",
        style="Note.TLabel", wraplength=WRAP, justify="left"
    ).grid(row=6, column=0, columnspan=3, sticky="we", pady=(0, 4))

    ttk.Checkbutton(
        opts, text="Clean up previous run's intermediate files", variable=cleanup_var
    ).grid(row=7, column=0, columnspan=3, sticky="w", pady=4)

    # --------------------------------------------------------
    # Run / Cancel
    # --------------------------------------------------------

    result = {"ok": False}

    def on_run():
        try:
            scale_val = float(scale_var.get())
        except ValueError:
            messagebox.showerror("Error", "Drizzle scale must be a number.")
            return

        if scale_val < 1.0 or scale_val > 3.0:
            messagebox.showerror("Error", "Drizzle scale must be between 1.0 and 3.0.")
            return

        try:
            pixfrac_val = float(pixfrac_var.get())
        except ValueError:
            messagebox.showerror("Error", "Drizzle pixel fraction must be a number.")
            return

        if pixfrac_val <= 0 or pixfrac_val > 1.0:
            messagebox.showerror("Error", "Drizzle pixel fraction must be between 0 (exclusive) and 1.0.")
            return

        resolved = []
        for sv in sessions:
            filter_type = sv["filter_var"].get().strip() or FILTER_TYPES[0]
            lights_text = sv["lights_var"].get().strip()
            lights = Path(lights_text) if lights_text else None
            if lights is not None and has_files(lights):
                resolved.append({
                    "filter": filter_type,
                    "lights": lights,
                    "darks": Path(sv["darks_var"].get().strip()) if sv["darks_enabled"].get() and sv["darks_var"].get().strip() else None,
                    "flats": Path(sv["flats_var"].get().strip()) if sv["flats_enabled"].get() and sv["flats_var"].get().strip() else None,
                    "biases": Path(sv["biases_var"].get().strip()) if sv["biases_enabled"].get() and sv["biases_var"].get().strip() else None,
                })

        if not resolved:
            messagebox.showerror(
                "Error",
                "At least one session needs a valid, non-empty Lights folder."
            )
            return

        result.update(
            ok=True,
            scale=scale_val,
            pixfrac=pixfrac_val,
            cleanup=cleanup_var.get(),
            match_backgrounds=match_backgrounds_var.get(),
            weight=WEIGHT_LABEL_TO_VALUE[weight_var.get()],
            sessions=resolved,
        )
        win.destroy()

    def on_cancel():
        result["ok"] = False
        win.destroy()

    btns = ttk.Frame(outer)
    btns.grid(row=3, column=0, columnspan=2, pady=(10, 0))
    ttk.Button(btns, text="Run", command=on_run).pack(side="left", padx=5)
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=5)

    win.protocol("WM_DELETE_WINDOW", on_cancel)
    win.mainloop()

    return result if result["ok"] else None


def main():

    siril = s.SirilInterface()

    try:
        siril.connect()
    except SirilConnectionError as exc:
        print(f"Unable to connect to Siril: {exc}")
        return

    try:

        siril.cmd("requires", "1.4.0")

        root = Path(siril.get_siril_wd())

        # ----------------------------------------------------
        # Resolve settings: GUI or headless
        # ----------------------------------------------------

        if USE_GUI:

            settings = prompt_settings(root)

            if settings is None:
                siril.log("Cancelled by user.")
                return

            scale = settings["scale"]
            pixfrac = settings["pixfrac"]
            cleanup_previous = settings["cleanup"]
            match_backgrounds = settings["match_backgrounds"]
            stack_weight = settings["weight"]
            session_settings = settings["sessions"]

        else:

            scale = SCALE
            pixfrac = PIXFRAC
            cleanup_previous = CLEANUP_PREVIOUS
            match_backgrounds = MATCH_BACKGROUNDS
            stack_weight = STACK_WEIGHT

            session_settings = []
            for entry in SESSIONS:
                filter_type = entry.get("filter")
                matched = next(
                    (f for f in FILTER_TYPES if f.lower() == (filter_type or "").strip().lower()),
                    None
                )
                if matched is None:
                    raise ValueError(
                        f'SESSIONS filter {filter_type!r} must be one of {FILTER_TYPES}'
                    )
                lights_dir, darks_dir, flats_dir, biases_dir = default_session_dirs(root, matched, entry)
                session_settings.append({
                    "filter": matched,
                    "lights": lights_dir,
                    "darks": darks_dir,
                    "flats": flats_dir,
                    "biases": biases_dir,
                })

        if scale < 1.0 or scale > 3.0:
            raise ValueError("Scale must be between 1.0 and 3.0")

        if pixfrac <= 0 or pixfrac > 1.0:
            raise ValueError("PIXFRAC must be between 0 (exclusive) and 1.0")

        if stack_weight is not None and stack_weight not in STACK_WEIGHT_CHOICES:
            raise ValueError(f"Invalid STACK_WEIGHT: {stack_weight!r}")

        scale_is_one = abs(scale - 1.0) < 1e-6
        scale_tag = f"{scale:g}".replace(".", "p")

        stack_weight_args = [f"-weight={stack_weight}"] if stack_weight else []

        # Dedicated temporary folders so we don't touch the normal
        # Siril process/masters directories.
        process = root / "_hao3_process"
        masters = root / "_hao3_masters"

        # Final results are saved directly into the current (Siril
        # working) folder.
        results = root

        # ----------------------------------------------------
        # Helpers
        # ----------------------------------------------------

        def log(message):
            print(message)
            siril.log(message)

        def cd(directory):
            siril.cmd("cd", str(directory))

        # ----------------------------------------------------
        # Which sessions are active? Build a unique key per session
        # (index + filter slug, so two sessions sharing a filter
        # don't collide on disk), and derive each session's red
        # channel identity ("Ha"/"SII") from its filter type.
        # ----------------------------------------------------

        sessions = []

        for i, entry in enumerate(session_settings):
            if has_files(entry["lights"]):
                sessions.append({
                    "key": f"{i}_{slugify(entry['filter'])}",
                    "label": FILTER_TO_CHANNEL[entry["filter"]],
                    "lights": entry["lights"],
                    "darks": entry["darks"],
                    "flats": entry["flats"],
                    "biases": entry["biases"],
                })

        if not sessions:
            raise RuntimeError(
                "No light frames found. Provide a non-empty lights "
                "folder for at least one session."
            )

        log("==========================================")
        log("OSC narrowband preprocess (multi-session)")
        log("==========================================")

        for session in sessions:
            n = len(list_light_files(session["lights"]))
            log(f"[{session['label']}] Lights : {session['lights']} ({n} frames)")
            log(f"[{session['label']}] Darks  : {session['darks'] if has_files(session['darks']) else 'NO'}")
            log(f"[{session['label']}] Flats  : {session['flats'] if has_files(session['flats']) else 'NO'}")
            log(f"[{session['label']}] Biases : {session['biases'] if has_files(session['biases']) else 'NO'}")

        log(f"Scale  : {scale:g}")
        log(f"Weight : {stack_weight or 'OFF'}")
        log(f"Cleanup: {'YES' if cleanup_previous else 'NO (reusing previous run)'}")
        log("==========================================")

        # ----------------------------------------------------
        # Reset script-owned temporary folders
        # ----------------------------------------------------

        if cleanup_previous:

            if process.exists():
                shutil.rmtree(process)

            if masters.exists():
                shutil.rmtree(masters)

        process.mkdir(exist_ok=True)
        masters.mkdir(exist_ok=True)

        def merge_sequence_files(sources, dest_dir, dest_prefix):
            """
            Move every file matching "<sequence_prefix>_*.fit" from
            each (process_dir, sequence_prefix) source into dest_dir,
            renumbered sequentially as "<dest_prefix>_NNNNN.fit".
            Returns the number of frames moved.
            """
            dest_dir.mkdir(parents=True, exist_ok=True)
            idx = 0
            for src_dir, sequence_prefix in sources:
                for f in sorted(src_dir.glob(f"{sequence_prefix}_*.fit")):
                    idx += 1
                    shutil.move(str(f), str(dest_dir / f"{dest_prefix}_{idx:05d}.fit"))
            return idx

        # ----------------------------------------------------
        # Per-session processing: calibrate + convert lights, extract
        # red channel + OIII. Registration/drizzle/stacking happens
        # afterward, grouped by label (see below) - sessions sharing
        # a label (e.g. two "Ha" sessions shot on different nights)
        # get their red channel merged into ONE combined stack, the
        # same way OIII always merges across every session regardless
        # of label.
        # ----------------------------------------------------

        def process_session(session):

            key = session["key"]
            label = session["label"]
            lights_dir = session["lights"]
            darks_dir = session["darks"]
            flats_dir = session["flats"]
            biases_dir = session["biases"]

            session_process = process / key
            session_masters = masters / key
            session_process.mkdir(parents=True, exist_ok=True)
            session_masters.mkdir(parents=True, exist_ok=True)

            has_biases = has_files(biases_dir)
            has_darks = has_files(darks_dir)
            has_flats = has_files(flats_dir)

            def convert_frames(kind, frames_dir):
                cd(frames_dir)
                siril.cmd("convert", kind, f"-out={session_process}")
                cd(session_process)

            def build_simple_master(kind, has_frames, frames_dir):
                # Shared by BIAS and DARKS, which are identical apart
                # from the frame kind - FLATS additionally needs the
                # bias-calibration branch below, so it stays separate.
                if not has_frames:
                    log(f"[{label}] No {kind}s found -> skipping master {kind}.")
                    return
                log(f"[{label}] Creating master {kind}...")
                convert_frames(kind, frames_dir)
                siril.cmd(
                    "stack", kind,
                    "rej", "3", "3",
                    "-nonorm", "-32b",
                    f"-out={session_masters / f'{kind}_stacked'}"
                )

            # ---- BIAS ----

            build_simple_master("bias", has_biases, biases_dir)

            # ---- FLATS ----

            if has_flats:

                log(f"[{label}] Creating master flat...")
                convert_frames("flat", flats_dir)

                if has_biases:
                    log(f"[{label}] Calibrating flats with master bias.")
                    siril.cmd(
                        "calibrate", "flat",
                        f"-bias={session_masters / 'bias_stacked'}"
                    )
                    flat_sequence = "pp_flat"
                else:
                    log(
                        f"[{label}] WARNING: No bias frames available. "
                        "Flats will be stacked without bias calibration."
                    )
                    flat_sequence = "flat"

                siril.cmd(
                    "stack", flat_sequence,
                    "rej", "3", "3",
                    "-norm=mul", "-32b",
                    f"-out={session_masters / 'flat_stacked'}"
                )

            else:

                log(f"[{label}] No flats found -> skipping flat correction.")

            # ---- DARKS ----

            build_simple_master("dark", has_darks, darks_dir)

            # ---- LIGHTS: convert + calibrate ----

            log(f"[{label}] Converting lights...")
            convert_frames("light", lights_dir)

            calibration_args = []

            # If a dark exists, do NOT also subtract bias from the
            # lights - the master dark already contains the offset.
            if has_darks:
                calibration_args.extend([
                    f"-dark={session_masters / 'dark_stacked'}",
                    "-cc=dark"
                ])
                if has_biases:
                    log(
                        f"[{label}] Master bias will NOT be applied "
                        "separately to lights because a master dark "
                        "is present."
                    )
            elif has_biases:
                calibration_args.append(
                    f"-bias={session_masters / 'bias_stacked'}"
                )

            if has_flats:
                calibration_args.extend([
                    f"-flat={session_masters / 'flat_stacked'}",
                    "-equalize_cfa"
                ])

            calibration_args.append("-cfa")

            if has_biases or has_darks or has_flats:

                methods = []
                if has_darks:
                    methods.append("dark")
                elif has_biases:
                    methods.append("bias")
                if has_flats:
                    methods.append("flat")

                log(f"[{label}] Calibrating lights using: " + ", ".join(methods))

                siril.cmd("calibrate", "light", *calibration_args)

                light_sequence = "pp_light"

            else:

                log(f"[{label}] No calibration frames found -> using converted lights directly.")

                light_sequence = "light"

            # ---- EXTRACT red channel + OIII ----

            log(f"[{label}] Extracting {label} and OIII...")

            # Deliberately DO NOT use -resample=ha: the red-channel
            # extraction stays half-resolution so drizzle gets the
            # original red CFA samples.
            siril.cmd("seqextract_HaOIII", light_sequence)

            red_sequence = f"Ha_{light_sequence}"
            oiii_sequence = f"OIII_{light_sequence}"

            return {
                "light_files": list_light_files(lights_dir),
                "process_dir": session_process,
                "red_sequence": red_sequence,
                "oiii_sequence": oiii_sequence,
            }

        results_by_key = {}

        for session in sessions:
            log("------------------------------------------")
            log(f"Processing {session['label']} session")
            log("------------------------------------------")
            results_by_key[session["key"]] = process_session(session)

        # ----------------------------------------------------
        # Group sessions by their red-channel label ("Ha"/"SII",
        # always exactly one of those two - see FILTER_TO_CHANNEL):
        # every group's red-channel frames (across all its member
        # sessions) are merged into one sequence and registered/
        # drizzled/stacked once, giving one combined red-channel
        # master per label instead of one per session.
        # ----------------------------------------------------

        groups = {}

        for session in sessions:
            groups.setdefault(session["label"], []).append(session["key"])

        channel_results = {}

        for label, keys in groups.items():

            member_infos = [results_by_key[k] for k in keys]

            log("------------------------------------------")
            log(f"Building combined {label} stack ({len(member_infos)} session(s))")
            log("------------------------------------------")

            red_dir = process / f"red_{slugify(label)}"
            sources = [(info["process_dir"], info["red_sequence"]) for info in member_infos]
            count = merge_sequence_files(sources, red_dir, "red_all")
            log(f"Combined {label} frame count: {count}")

            cd(red_dir)

            log(f"Calculating {label} registration...")
            siril.cmd("register", "red_all", "-2pass")

            log(f"Applying true 2x drizzle to {label}...")
            siril.cmd(
                "seqapplyreg", "red_all",
                "-scale=2", "-drizzle",
                f"-pixfrac={pixfrac}", f"-kernel={KERNEL}",
                "-framing=min"
            )

            log(f"Stacking {label}...")
            siril.cmd(
                "stack", "r_red_all",
                "rej", "3", "3",
                "-norm=addscale",
                *stack_weight_args,
                "-output_norm", "-32b",
                "-out=red_native"
            )

            siril.cmd("mirrorx_single", "red_native")
            siril.cmd("load", "red_native")

            if scale_is_one:
                log(f"SCALE = 1 -> {label} already at final resolution.")
            else:
                log(f"SCALE = {scale:g} -> Lanczos resampling {label} to final resolution.")
                siril.cmd("resample", f"{scale:g}", "-interp=lanczos4")

            siril.cmd("save", "red_final")

            light_files_all = []
            for info in member_infos:
                light_files_all.extend(info["light_files"])

            channel_results[label] = {
                "light_files": light_files_all,
                "red_final_path": red_dir / "red_final.fit",
            }

        # ----------------------------------------------------
        # Combine OIII across every session into one sequence and
        # stack it once - this is what gives the deeper/better SNR
        # combined OIII master when multiple sessions are provided.
        # ----------------------------------------------------

        log("------------------------------------------")
        log("Combining OIII from all sessions")
        log("------------------------------------------")

        oiii_dir = process / "oiii_combined"
        oiii_sources = [(info["process_dir"], info["oiii_sequence"]) for info in results_by_key.values()]
        frame_idx = merge_sequence_files(oiii_sources, oiii_dir, "oiii_all")
        oiii_light_files_all = [f for info in results_by_key.values() for f in info["light_files"]]

        log(f"Combined OIII frame count: {frame_idx}")

        cd(oiii_dir)

        log("Calculating combined OIII registration...")
        siril.cmd("register", "oiii_all", "-2pass")

        if scale_is_one:
            log("SCALE = 1 -> applying normal OIII registration.")
            siril.cmd("seqapplyreg", "oiii_all", "-interp=lanczos4", "-framing=min")
        else:
            log(f"Applying true {scale:g}x drizzle to OIII...")
            siril.cmd(
                "seqapplyreg", "oiii_all",
                f"-scale={scale:g}", "-drizzle",
                f"-pixfrac={pixfrac}", f"-kernel={KERNEL}",
                "-framing=min"
            )

        log("Stacking combined OIII...")
        siril.cmd(
            "stack", "r_oiii_all",
            "rej", "3", "3",
            "-norm=addscale",
            *stack_weight_args,
            "-output_norm", "-32b",
            "-out=oiii_native"
        )

        siril.cmd("mirrorx_single", "oiii_native")

        oiii_final_path = oiii_dir / "oiii_native.fit"

        # ----------------------------------------------------
        # Align every final channel (one per unique label + combined
        # OIII) together on one shared pixel grid.
        # ----------------------------------------------------

        log("------------------------------------------")
        log("Aligning all final channel masters...")
        log("------------------------------------------")

        final_dir = process / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        channel_indices = {}
        idx = 0

        for label, info in channel_results.items():
            idx += 1
            channel_indices[label] = idx
            shutil.move(
                str(info["red_final_path"]),
                str(final_dir / f"final_{idx:05d}.fit")
            )

        idx += 1
        oiii_idx = idx
        shutil.move(str(oiii_final_path), str(final_dir / f"final_{idx:05d}.fit"))

        cd(final_dir)

        #
        # -2pass here only computes the shift; seqapplyreg then crops
        # every final_NNNNN to their mutual common area (-framing=min),
        # which is required for them to come out the same pixel size -
        # they were cropped independently (each to its own sequence's
        # overlap) during the per-channel/combined stacks above, so
        # their sizes can differ by a few pixels before this step.
        #

        siril.cmd("register", "final", "-2pass", "-transf=shift")
        siril.cmd("seqapplyreg", "final", "-interp=lanczos4", "-framing=min")

        # ----------------------------------------------------
        # Background-only match: every non-reference channel gets
        # its median (background level) shifted to equal the
        # reference channel's median. Contrast/noise (MAD) is left
        # untouched on every channel, including the reference, so no
        # channel's noise gets amplified by this step - it only
        # gives every channel the same black point for a clean
        # composite. Reference = "Ha" if present, else the first
        # channel.
        # ----------------------------------------------------

        reference_label = "Ha" if "Ha" in channel_results else next(iter(channel_results))
        reference_idx = channel_indices[reference_label]

        def median_match_expression(source_idx):
            return (
                f"$r_final_{source_idx:05d}$"
                f"-median($r_final_{source_idx:05d}$)"
                f"+median($r_final_{reference_idx:05d}$)"
            )

        if match_backgrounds:
            log(f"Matching backgrounds to {reference_label} (median only, no contrast scaling)...")
        else:
            log("Background matching disabled -> saving each channel as stacked.")

        def save_channel(idx, is_reference, out_path):
            if match_backgrounds and not is_reference:
                siril.cmd("pm", f'"{median_match_expression(idx)}"')
            else:
                siril.cmd("load", f"r_final_{idx:05d}")
            siril.cmd("save", str(out_path))

        oiii_name = f"{derive_base_name_from_files(oiii_light_files_all)}_OIII_x{scale_tag}"
        save_channel(oiii_idx, is_reference=False, out_path=results / oiii_name)

        output_names = {"OIII": oiii_name}

        for label, idx in channel_indices.items():
            info = channel_results[label]
            name = f"{derive_base_name_from_files(info['light_files'])}_{label}_x{scale_tag}"
            output_names[label] = name
            save_channel(idx, is_reference=(label == reference_label), out_path=results / name)

        # ====================================================
        # DONE
        # ====================================================

        siril.cmd("close")

        # Move Siril's CWD out of _hao3_process before deleting it -
        # otherwise Windows can refuse to remove a directory that is
        # still a process's current working directory.
        cd(root)

        if cleanup_previous:

            log("Cleaning up temporary folders...")

            if process.exists():
                shutil.rmtree(process)

            if masters.exists():
                shutil.rmtree(masters)

        log("==========================================")
        log("Finished successfully")
        log(f"SCALE = {scale:g}")
        log("Results: " + ", ".join(output_names.values()))
        log("==========================================")

    except Exception as exc:

        message = f"Narrowband preprocessing failed:\n{exc}"

        print(message)

        try:
            siril.log(message)
            siril.error_messagebox(message)
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()
