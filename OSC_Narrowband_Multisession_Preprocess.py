# OSC_Narrowband_Multisession_Preprocess.py
#
# Siril 1.4+
#
# OSC dual-band narrowband preprocessing with an arbitrary number of
# sessions (not just a fixed Ha-OIII / SII-OIII pair - add as many
# sessions as you have, each labeled with its own red-channel line,
# e.g. "Ha", "SII", or a custom name). Every session's OIII gets
# merged into ONE combined sequence and stacked once (deeper/better
# SNR than any single session's OIII alone); every session's red
# channel is stacked on its own. All resulting channels (N red +
# 1 combined OIII) are then aligned together on a shared pixel grid.
#
# Session concept and UI approach adapted from Naztronomy's OSC
# preprocessing script (https://github.com/naztronaut/siril-scripts),
# reimplemented in plain tkinter (no extra dependencies) instead of
# PyQt6.
#
# Siril's seqextract_HaOIII always splits Bayer CFA data into a
# "red channel" sequence and a "green+blue channel" sequence and
# always names them Ha_*/OIII_* internally, regardless of which real
# narrowband line the red channel represents - a session's "label"
# below only controls how ITS output file is named (and its default
# folder names); the extraction itself is identical for every session.
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
#   lights_<label>/    REQUIRED (per session; skip a session with an
#                       empty/missing lights folder)
#   darks_<label>/     optional
#   flats_<label>/     optional
#   biases_<label>/    optional
#
# where <label> is the session's label, lowercased and stripped to
# letters/digits only (e.g. label "Ha" -> lights_ha).
#
# OUTPUT:
#
# Saved directly into Siril's current working folder. Each session's
# red channel gets its own name built from ITS light frames (common
# filename prefix, kept whole through a shared date even though
# per-frame time differs, plus "<n>x<exposure>s" or "<n>f"); the
# combined OIII's name is built the same way from EVERY session's
# light frames put together:
#
#   <session lights prefix>_<label>_x<scale>.fit   (one per session)
#   <combined lights prefix>_OIII_x<scale>.fit      (from ALL sessions)
#
#
# NORMALIZATION
# -------------
#
# Every non-reference channel gets its background level (median)
# shifted to match the reference channel's median, via PixelMath.
# The reference is the first session labeled "Ha" (case-insensitive)
# if any, otherwise the first session in the list. Contrast/noise
# (MAD) is left untouched on every channel - this only gives every
# channel the same black point for a clean composite, without
# amplifying a fainter channel's noise the way matching contrast/
# scale would. It is not a scientific calibration. The reference
# channel itself is saved as-is, un-normalized.
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

# One entry per session. "label" names the session's red channel in
# the output filename (and, if the folder fields below are left as
# None, its default folder names - see INPUT FOLDERS above). Add or
# remove entries freely; a session with no light frames is skipped.
SESSIONS = [
    {"label": "Ha", "lights": None, "darks": None, "flats": None, "biases": None},
    {"label": "SII", "lights": None, "darks": None, "flats": None, "biases": None},
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


def resolve_dir(value, root, name):
    return Path(value) if value else (root / name)


def slugify(label):
    """Folder-name-safe version of a session label, e.g. "Ha" -> "ha"."""
    slug = re.sub(r"[^A-Za-z0-9]+", "", label or "").lower()
    return slug or "session"


def sanitize_token(text):
    """
    Collapse whitespace to underscores. Siril's cmd() joins arguments
    with plain spaces (no quoting), so a space in a filename/sequence
    name would break commands like "save".
    """
    text = re.sub(r"\s+", "_", (text or "").strip())
    text = re.sub(r"_+", "_", text)
    return text or "session"


def default_session_dirs(root, label, overrides):
    """
    Resolve a session's 4 folder settings against their
    "<kind>_<slug(label)>" defaults under root, e.g. label="Ha" ->
    lights_ha, darks_ha, flats_ha, biases_ha.
    """
    slug = slugify(label)
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

    common = common.strip("_- ")
    common = sanitize_token(common)

    if not common or common == "session":
        common = "narrowband"

    suffix = build_exposure_suffix(stems, count)

    return f"{common}_{suffix}"


def prompt_settings(root):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    win = tk.Tk()
    win.title("Narrowband Preprocess (multi-session)")

    style = ttk.Style()
    style.configure("Header.TLabel", font=("TkDefaultFont", 9, "bold"))
    style.configure("Note.TLabel", font=("TkDefaultFont", 8), foreground="#666666")

    outer = ttk.Frame(win, padding=10)
    outer.grid(sticky="nsew")

    scale_var = tk.StringVar(value=f"{SCALE:g}")
    pixfrac_var = tk.StringVar(value=f"{PIXFRAC:g}")
    cleanup_var = tk.BooleanVar(value=CLEANUP_PREVIOUS)
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

    def make_session_vars(label, lights_d, darks_d, flats_d, biases_d):
        return {
            "label_var": tk.StringVar(value=label),
            "lights_var": tk.StringVar(value=str(lights_d)),
            "darks_var": tk.StringVar(value=str(darks_d)),
            "darks_enabled": tk.BooleanVar(value=has_files(darks_d)),
            "flats_var": tk.StringVar(value=str(flats_d)),
            "flats_enabled": tk.BooleanVar(value=has_files(flats_d)),
            "biases_var": tk.StringVar(value=str(biases_d)),
            "biases_enabled": tk.BooleanVar(value=has_files(biases_d)),
        }

    for entry in SESSIONS:
        label = entry.get("label") or f"Session {len(sessions) + 1}"
        dirs = default_session_dirs(root, label, entry)
        sessions.append(make_session_vars(label, *dirs))

    if not sessions:
        sessions.append(make_session_vars("Ha", *default_session_dirs(root, "Ha", {})))

    # --------------------------------------------------------
    # Left: session list + add/remove
    # --------------------------------------------------------

    left = ttk.Frame(outer)
    left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

    ttk.Label(left, text="Sessions", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")

    listbox = tk.Listbox(left, width=32, height=10, exportselection=False)
    listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(2, 4))

    def describe(index):
        sv = sessions[index]
        label = sv["label_var"].get() or f"Session {index + 1}"
        lights_text = sv["lights_var"].get()
        n = len(list_light_files(Path(lights_text))) if lights_text else 0
        return f"{index + 1}: {label} ({n} lights)"

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
        n = len(sessions) + 1
        label = f"Session {n}"
        sessions.append(make_session_vars(label, *default_session_dirs(root, label, {})))
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
    editor.grid(row=0, column=1, sticky="nsew")

    def set_state(widget, enabled):
        widget.state(["!disabled"] if enabled else ["disabled"])

    erow = 0

    ttk.Label(editor, text="Channel label:").grid(row=erow, column=0, sticky="w")
    label_entry = ttk.Entry(editor, width=20)
    label_entry.grid(row=erow, column=1, sticky="w")
    erow += 1

    ttk.Label(
        editor,
        text='Names this session\'s red-channel output and its default '
             'folders (lights_<label>, darks_<label>, ...). Use "Ha" for '
             'a Ha-OIII filter, "SII" for SII-OIII, or anything else for '
             "a custom line.",
        style="Note.TLabel", wraplength=340, justify="left"
    ).grid(row=erow, column=0, columnspan=3, sticky="w", pady=(0, 8))
    erow += 1

    def browse_into(var):
        path = filedialog.askdirectory(initialdir=var.get() or str(root))
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

        label_entry.configure(textvariable=sv["label_var"])
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

    # Re-describe the listbox row when the label text changes, so it
    # doesn't go stale while editing.
    for sv in sessions:
        sv["label_var"].trace_add("write", on_label_change)

    # --------------------------------------------------------
    # Shared options
    # --------------------------------------------------------

    opts = ttk.Frame(outer)
    opts.grid(row=1, column=0, columnspan=2, sticky="we", pady=(10, 0))

    ttk.Label(opts, text="Options", style="Header.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")

    ttk.Label(opts, text="Drizzle scale:").grid(row=1, column=0, sticky="w", pady=(4, 0))
    ttk.Combobox(
        opts, textvariable=scale_var, values=["1", "1.5", "2", "2.5", "3"], width=8
    ).grid(row=1, column=1, sticky="w", pady=(4, 0))

    ttk.Label(
        opts,
        text="Note: every session's red channel is always true-drizzled "
             "2x regardless of this value; it only adds further scaling on top.",
        style="Note.TLabel", wraplength=420, justify="left"
    ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(0, 4))

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
        opts, text="Clean up previous run's intermediate files", variable=cleanup_var
    ).grid(row=5, column=0, columnspan=3, sticky="w", pady=4)

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
            label = sv["label_var"].get().strip() or "Session"
            lights_text = sv["lights_var"].get().strip()
            lights = Path(lights_text) if lights_text else None
            if lights is not None and has_files(lights):
                resolved.append({
                    "label": label,
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
            weight=WEIGHT_LABEL_TO_VALUE[weight_var.get()],
            sessions=resolved,
        )
        win.destroy()

    def on_cancel():
        result["ok"] = False
        win.destroy()

    btns = ttk.Frame(outer)
    btns.grid(row=2, column=0, columnspan=2, pady=(10, 0))
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
            stack_weight = settings["weight"]
            session_settings = settings["sessions"]

        else:

            scale = SCALE
            pixfrac = PIXFRAC
            cleanup_previous = CLEANUP_PREVIOUS
            stack_weight = STACK_WEIGHT

            session_settings = []
            for entry in SESSIONS:
                label = entry.get("label") or f"Session {len(session_settings) + 1}"
                lights_dir, darks_dir, flats_dir, biases_dir = default_session_dirs(root, label, entry)
                session_settings.append({
                    "label": label,
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
        # (index + label slug, so two sessions sharing a label don't
        # collide on disk).
        # ----------------------------------------------------

        sessions = []

        for i, entry in enumerate(session_settings):
            if has_files(entry["lights"]):
                sessions.append({
                    "key": f"{i}_{slugify(entry['label'])}",
                    "label": sanitize_token(entry["label"]),
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

        # ----------------------------------------------------
        # Per-session processing: calibrate + convert lights,
        # extract red channel + OIII, register/drizzle/stack the
        # red channel. OIII extraction is left un-stacked here - it
        # gets merged across sessions and stacked once, below.
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

            # ---- BIAS ----

            if has_biases:
                log(f"[{label}] Creating master bias...")
                convert_frames("bias", biases_dir)
                siril.cmd(
                    "stack", "bias",
                    "rej", "3", "3",
                    "-nonorm", "-32b",
                    f"-out={session_masters / 'bias_stacked'}"
                )
            else:
                log(f"[{label}] No biases found -> skipping master bias.")

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

            if has_darks:
                log(f"[{label}] Creating master dark...")
                convert_frames("dark", darks_dir)
                siril.cmd(
                    "stack", "dark",
                    "rej", "3", "3",
                    "-nonorm", "-32b",
                    f"-out={session_masters / 'dark_stacked'}"
                )
            else:
                log(f"[{label}] No darks found -> skipping dark correction.")

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

            # ---- RED CHANNEL: register + true 2x drizzle + stack ----

            log(f"[{label}] Calculating {label} registration...")
            siril.cmd("register", red_sequence, "-2pass")

            log(f"[{label}] Applying true 2x drizzle to {label}...")
            siril.cmd(
                "seqapplyreg", red_sequence,
                "-scale=2", "-drizzle",
                f"-pixfrac={pixfrac}", f"-kernel={KERNEL}",
                "-framing=min"
            )

            registered_red = f"r_{red_sequence}"

            log(f"[{label}] Stacking {label}...")
            siril.cmd(
                "stack", registered_red,
                "rej", "3", "3",
                "-norm=addscale",
                *stack_weight_args,
                "-output_norm", "-32b",
                "-out=red_native"
            )

            siril.cmd("mirrorx_single", "red_native")
            siril.cmd("load", "red_native")

            if scale_is_one:
                log(f"[{label}] SCALE = 1 -> {label} already at final resolution.")
            else:
                log(f"[{label}] SCALE = {scale:g} -> Lanczos resampling {label} to final resolution.")
                siril.cmd("resample", f"{scale:g}", "-interp=lanczos4")

            siril.cmd("save", "red_final")

            return {
                "key": key,
                "label": label,
                "light_files": list_light_files(lights_dir),
                "red_final_path": session_process / "red_final.fit",
                "oiii_sequence": oiii_sequence,
                "oiii_process_dir": session_process,
            }

        results_by_key = {}

        for session in sessions:
            log("------------------------------------------")
            log(f"Processing {session['label']} session")
            log("------------------------------------------")
            results_by_key[session["key"]] = process_session(session)

        # ----------------------------------------------------
        # Combine OIII across every session into one sequence and
        # stack it once - this is what gives the deeper/better SNR
        # combined OIII master when multiple sessions are provided.
        # ----------------------------------------------------

        log("------------------------------------------")
        log("Combining OIII from all sessions")
        log("------------------------------------------")

        oiii_dir = process / "oiii_combined"
        oiii_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        oiii_light_files_all = []

        for info in results_by_key.values():
            src_files = sorted(
                info["oiii_process_dir"].glob(f"{info['oiii_sequence']}_*.fit")
            )
            for f in src_files:
                frame_idx += 1
                shutil.move(str(f), str(oiii_dir / f"oiii_all_{frame_idx:05d}.fit"))
            oiii_light_files_all.extend(info["light_files"])

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
        # Align every final channel (N red channels + combined OIII)
        # together on one shared pixel grid.
        # ----------------------------------------------------

        log("------------------------------------------")
        log("Aligning all final channel masters...")
        log("------------------------------------------")

        final_dir = process / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        channel_indices = {}
        idx = 0

        for session in sessions:
            idx += 1
            channel_indices[session["key"]] = idx
            shutil.move(
                str(results_by_key[session["key"]]["red_final_path"]),
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
        # composite. Reference = the first session labeled "Ha"
        # (case-insensitive) if any, else the first session in the
        # list.
        # ----------------------------------------------------

        reference_key = next(
            (s["key"] for s in sessions if s["label"].lower() == "ha"),
            sessions[0]["key"]
        )
        reference_label = results_by_key[reference_key]["label"]
        reference_idx = channel_indices[reference_key]

        def median_match_expression(source_idx):
            return (
                f"$r_final_{source_idx:05d}$"
                f"-median($r_final_{source_idx:05d}$)"
                f"+median($r_final_{reference_idx:05d}$)"
            )

        log(f"Matching backgrounds to {reference_label} (median only, no contrast scaling)...")

        siril.cmd("pm", f'"{median_match_expression(oiii_idx)}"')

        oiii_name = f"{derive_base_name_from_files(oiii_light_files_all)}_OIII_x{scale_tag}"
        siril.cmd("save", str(results / oiii_name))

        output_names = {"oiii": oiii_name}

        for key, idx in channel_indices.items():
            info = results_by_key[key]

            if key == reference_key:
                siril.cmd("load", f"r_final_{idx:05d}")
            else:
                siril.cmd("pm", f'"{median_match_expression(idx)}"')

            name = f"{derive_base_name_from_files(info['light_files'])}_{info['label']}_x{scale_tag}"
            output_names[key] = name
            siril.cmd("save", str(results / name))

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
