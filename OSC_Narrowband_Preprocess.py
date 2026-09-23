# OSC_Narrowband_Preprocess.py
#
# Siril 1.4+
#
# OSC dual-band narrowband preprocessing. Supports up to two sessions
# at once - a Ha-OIII dual-band filter session and/or an SII-OIII
# dual-band filter session - and produces up to 3 final channel
# masters: Ha, SII, and a single combined OIII built from every OIII
# sub across BOTH sessions (deeper/better SNR than either session's
# OIII alone). Provide at least one session; the other is skipped if
# its lights folder is empty/absent.
#
# Siril's seqextract_HaOIII always splits Bayer CFA data into a
# "red channel" sequence and a "green+blue channel" sequence and
# always names them Ha_*/OIII_* internally, regardless of which real
# narrowband line the red channel represents - SII sits on the same
# red Bayer pixels Ha would, so the same extraction works for both
# filter types; only the output filenames differ (Ha vs SII).
#
# Can be run two ways:
#
#   USE_GUI = True   -> shows a dialog to pick everything below,
#                        for interactive use from Siril's GUI.
#   USE_GUI = False  -> uses the USER SETTING values as-is, for
#                        headless / scripted use.
#
# INPUT FOLDERS (headless mode; defaults under Siril's working dir
# unless HA_*/SII_* overrides are set):
#
#   lights_haoiii/    Ha-OIII session lights (skip session if empty)
#   darks_haoiii/     optional
#   flats_haoiii/     optional
#   biases_haoiii/    optional
#
#   lights_siioiii/   SII-OIII session lights (skip session if empty)
#   darks_siioiii/    optional
#   flats_siioiii/    optional
#   biases_siioiii/   optional
#
# OUTPUT:
#
# Saved directly into Siril's current working folder. Each present
# channel gets its own name built from ITS session's light frames
# (common filename prefix, kept whole through a shared date even
# though per-frame time differs, plus "<n>x<exposure>s" or "<n>f"):
#
#   <ha lights prefix>_Ha_x<scale>.fit        (if Ha-OIII given)
#   <sii lights prefix>_SII_x<scale>.fit      (if SII-OIII given)
#   <combined lights prefix>_OIII_x<scale>.fit  (from ALL OIII subs)
#
# e.g. with both sessions, 17x300s Ha-OIII lights and 12x600s
# SII-OIII lights (29 OIII subs total) ->
#   Unknown_HaOIII_20260919_17x300s_Ha_x1.fit
#   Unknown_SIIOIII_20260921_12x600s_SII_x1.fit
#   Unknown_20260919_29x300s+12x600s_OIII_x1.fit   (whatever prefix
#                                                     the two sessions'
#                                                     filenames share)
#
#
# NORMALIZATION
# -------------
#
# Every non-reference channel gets its background level (median)
# shifted to match the reference channel's median, via PixelMath.
# The reference is Ha if a Ha-OIII session was given, otherwise SII.
# Contrast/noise (MAD) is left untouched on every channel - this
# only gives every channel the same black point for a clean
# composite, without amplifying a fainter channel's noise the way
# matching contrast/scale would. It is not a scientific calibration.
# The reference channel itself is saved as-is, un-normalized.
#
#
# SCALE behaviour
# ---------------
#
# SCALE = 1
#
#   Ha / SII (each session's red channel):
#       half-res extraction
#       -> TRUE 2x drizzle
#       -> native full sensor resolution
#
#   OIII (combined across sessions):
#       native full-resolution extraction
#       -> normal registration
#
#
# SCALE > 1
#
#   Ha / SII:
#       half-res extraction
#       -> TRUE 2x drizzle
#       -> native full sensor resolution
#       -> Lanczos upscale by SCALE
#
#   OIII:
#       native full-resolution extraction
#       -> TRUE drizzle directly by SCALE
#
#
# ASI585 examples:
#
#   SCALE = 1.0  -> 3840 x 2160
#   SCALE = 1.5  -> 5760 x 3240
#   SCALE = 2.0  -> 7680 x 4320
#
# Siril drizzle supports scales up to 3.
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

# Drizzle scale factor (1.0 - 3.0). Note: the red channel (Ha/SII) is
# ALWAYS true-drizzled 2x regardless of this setting, since that's
# what reconstructs full sensor resolution from its half-res
# extraction - this setting adds further scaling on top of that (see
# SCALE behaviour in the header comment above).
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

# Provide at least one of the two sessions below. None = use
# <siril working dir>/<lights|darks|flats|biases>_haoiii (or
# _siioiii). Leave a session's LIGHTS dir unset with no matching
# default folder present to skip that session entirely.

HA_LIGHTS_DIR = None
HA_DARKS_DIR = None
HA_FLATS_DIR = None
HA_BIASES_DIR = None

SII_LIGHTS_DIR = None
SII_DARKS_DIR = None
SII_FLATS_DIR = None
SII_BIASES_DIR = None

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


def default_session_dirs(root, suffix, lights, darks, flats, biases):
    """
    Resolve a session's 4 folder settings (each either an explicit
    override or None) against their "<kind>_<suffix>" default under
    root, e.g. suffix="haoiii" -> lights_haoiii, darks_haoiii, ...
    """
    return (
        resolve_dir(lights, root, f"lights_{suffix}"),
        resolve_dir(darks, root, f"darks_{suffix}"),
        resolve_dir(flats, root, f"flats_{suffix}"),
        resolve_dir(biases, root, f"biases_{suffix}"),
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

    # Siril's cmd() joins arguments with plain spaces (no quoting), so
    # a space in the output filename would break commands like "save".
    common = re.sub(r"\s+", "_", common)
    common = re.sub(r"_+", "_", common)

    if not common:
        common = "narrowband"

    suffix = build_exposure_suffix(stems, count)

    return f"{common}_{suffix}"


def prompt_settings(root):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    win = tk.Tk()
    win.title("OSC Narrowband Preprocess")

    scale_var = tk.StringVar(value=f"{SCALE:g}")
    pixfrac_var = tk.StringVar(value=f"{PIXFRAC:g}")
    cleanup_var = tk.BooleanVar(value=CLEANUP_PREVIOUS)
    weight_var = tk.StringVar(value=STACK_WEIGHT_LABELS[STACK_WEIGHT])

    frm = ttk.Frame(win, padding=10)
    frm.grid(sticky="nsew")

    style = ttk.Style()
    style.configure("Header.TLabel", font=("TkDefaultFont", 9, "bold"))
    style.configure("Header.TCheckbutton", font=("TkDefaultFont", 9, "bold"))

    # Fixed wraplength wide enough to span the window's actual content
    # width (set by the wider entry/combobox rows below). A dynamic,
    # resize-driven wraplength was tried and removed: since this window
    # auto-sizes to fit its content, changing a label's wraplength on
    # <Configure> changes its requested size, which resizes the window,
    # which fires another <Configure> - an infinite resize loop.
    WRAP = 560

    row = 0

    def section(title):
        nonlocal row
        ttk.Label(frm, text=title, style="Header.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 2)
        )
        row += 1

    def set_state(widget, enabled):
        widget.state(["!disabled"] if enabled else ["disabled"])

    def build_session(label, lights_default, darks_default, flats_default, biases_default):
        nonlocal row

        session_var = tk.BooleanVar(value=has_files(lights_default))
        lights_var = tk.StringVar(value=str(lights_default))
        darks_enabled = tk.BooleanVar(value=has_files(darks_default))
        darks_var = tk.StringVar(value=str(darks_default))
        flats_enabled = tk.BooleanVar(value=has_files(flats_default))
        flats_var = tk.StringVar(value=str(flats_default))
        biases_enabled = tk.BooleanVar(value=has_files(biases_default))
        biases_var = tk.StringVar(value=str(biases_default))

        session_cb = ttk.Checkbutton(
            frm, text=f"{label}-OIII session", variable=session_var,
            style="Header.TCheckbutton"
        )
        session_cb.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 2))
        row += 1

        def folder_row(row_label, path_var, enabled_var=None):
            nonlocal row
            this_row = row

            if enabled_var is not None:
                cb = ttk.Checkbutton(frm, text=row_label, variable=enabled_var, command=lambda: refresh())
                cb.grid(row=this_row, column=0, sticky="w", padx=(20, 0))
            else:
                cb = None
                ttk.Label(frm, text=f"{row_label}:").grid(row=this_row, column=0, sticky="w", padx=(20, 0))

            entry = ttk.Entry(frm, textvariable=path_var, width=44)
            entry.grid(row=this_row, column=1, sticky="we")

            def browse():
                # Only start from the field's current value if it's a
                # real, existing folder (it's usually still a not-yet-
                # created default like ".../lights_haoiii") - otherwise
                # start from Siril's working directory (its "home").
                current = path_var.get()
                start = current if current and Path(current).is_dir() else str(root)
                path = filedialog.askdirectory(initialdir=start)
                if path:
                    path_var.set(path)

            btn = ttk.Button(frm, text="Browse...", command=browse)
            btn.grid(row=this_row, column=2)

            row += 1
            return cb, entry, btn

        lights_cb, lights_entry, lights_btn = folder_row("Lights", lights_var)
        darks_cb, darks_entry, darks_btn = folder_row("Darks", darks_var, darks_enabled)
        flats_cb, flats_entry, flats_btn = folder_row("Flats", flats_var, flats_enabled)
        biases_cb, biases_entry, biases_btn = folder_row("Biases", biases_var, biases_enabled)

        def refresh():
            enabled = session_var.get()
            set_state(lights_entry, enabled)
            set_state(lights_btn, enabled)
            for cb, entry, btn, sub_var in (
                (darks_cb, darks_entry, darks_btn, darks_enabled),
                (flats_cb, flats_entry, flats_btn, flats_enabled),
                (biases_cb, biases_entry, biases_btn, biases_enabled),
            ):
                set_state(cb, enabled)
                set_state(entry, enabled and sub_var.get())
                set_state(btn, enabled and sub_var.get())

        session_cb.configure(command=refresh)
        refresh()

        return {
            "session_var": session_var,
            "lights_var": lights_var,
            "darks_var": darks_var, "darks_enabled": darks_enabled,
            "flats_var": flats_var, "flats_enabled": flats_enabled,
            "biases_var": biases_var, "biases_enabled": biases_enabled,
        }

    ha = build_session("Ha", *default_session_dirs(
        root, "haoiii", HA_LIGHTS_DIR, HA_DARKS_DIR, HA_FLATS_DIR, HA_BIASES_DIR
    ))

    sii = build_session("SII", *default_session_dirs(
        root, "siioiii", SII_LIGHTS_DIR, SII_DARKS_DIR, SII_FLATS_DIR, SII_BIASES_DIR
    ))

    section("Options")

    style.configure("Note.TLabel", font=("TkDefaultFont", 8), foreground="#666666")

    ttk.Label(frm, text="Drizzle scale:").grid(row=row, column=0, sticky="w", pady=(4, 0))
    ttk.Combobox(
        frm, textvariable=scale_var, values=["1", "1.5", "2", "2.5", "3"],
        width=8
    ).grid(row=row, column=1, sticky="w", pady=(4, 0))
    row += 1

    ttk.Label(
        frm,
        text="The red channel is always 2x drizzled first, then scaled "
             "up further if this is set above 1. OIII follows this "
             "setting directly.",
        style="Note.TLabel", wraplength=WRAP, justify="left"
    ).grid(row=row, column=0, columnspan=3, sticky="we", pady=(0, 4))
    row += 1

    ttk.Label(frm, text="Drizzle pixel fraction:").grid(row=row, column=0, sticky="w", pady=4)
    ttk.Combobox(
        frm, textvariable=pixfrac_var, values=["0.5", "0.65", "0.8", "0.9", "1.0"],
        width=8
    ).grid(row=row, column=1, sticky="w", pady=4)
    row += 1

    ttk.Label(frm, text="Stack weighting:").grid(row=row, column=0, sticky="w", pady=4)
    ttk.Combobox(
        frm, textvariable=weight_var,
        values=list(STACK_WEIGHT_LABELS.values()),
        width=48, state="readonly"
    ).grid(row=row, column=1, columnspan=2, sticky="we", pady=4)
    row += 1

    ttk.Checkbutton(
        frm, text="Clean up previous run's intermediate files",
        variable=cleanup_var
    ).grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
    row += 1

    result = {"ok": False}

    def resolved_lights(session):
        if not session["session_var"].get():
            return None
        text = session["lights_var"].get().strip()
        path = Path(text) if text else None
        if path is None or not has_files(path):
            return "invalid"
        return path

    def cal_dir(session, key):
        if not session[f"{key}_enabled"].get():
            return None
        text = session[f"{key}_var"].get().strip()
        return Path(text) if text else None

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

        ha_lights = resolved_lights(ha)
        sii_lights = resolved_lights(sii)

        if ha_lights == "invalid":
            messagebox.showerror("Error", "Ha-OIII session is enabled but its Lights folder is empty or invalid.")
            return

        if sii_lights == "invalid":
            messagebox.showerror("Error", "SII-OIII session is enabled but its Lights folder is empty or invalid.")
            return

        if ha_lights is None and sii_lights is None:
            messagebox.showerror(
                "Error",
                "Enable at least one session (Ha-OIII and/or SII-OIII) "
                "with a valid Lights folder."
            )
            return

        result.update(
            ok=True,
            scale=scale_val,
            pixfrac=pixfrac_val,
            cleanup=cleanup_var.get(),
            weight=WEIGHT_LABEL_TO_VALUE[weight_var.get()],
            ha_lights_dir=ha_lights,
            ha_darks_dir=cal_dir(ha, "darks"),
            ha_flats_dir=cal_dir(ha, "flats"),
            ha_biases_dir=cal_dir(ha, "biases"),
            sii_lights_dir=sii_lights,
            sii_darks_dir=cal_dir(sii, "darks"),
            sii_flats_dir=cal_dir(sii, "flats"),
            sii_biases_dir=cal_dir(sii, "biases"),
        )
        win.destroy()

    def on_cancel():
        result["ok"] = False
        win.destroy()

    btns = ttk.Frame(frm)
    ttk.Button(btns, text="Run", command=on_run).pack(side="left", padx=5)
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=5)
    btns.grid(row=row, column=0, columnspan=3, pady=(10, 0))

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

            ha_lights_dir = settings["ha_lights_dir"]
            ha_darks_dir = settings["ha_darks_dir"]
            ha_flats_dir = settings["ha_flats_dir"]
            ha_biases_dir = settings["ha_biases_dir"]

            sii_lights_dir = settings["sii_lights_dir"]
            sii_darks_dir = settings["sii_darks_dir"]
            sii_flats_dir = settings["sii_flats_dir"]
            sii_biases_dir = settings["sii_biases_dir"]

        else:

            scale = SCALE
            pixfrac = PIXFRAC
            cleanup_previous = CLEANUP_PREVIOUS
            stack_weight = STACK_WEIGHT

            ha_lights_dir, ha_darks_dir, ha_flats_dir, ha_biases_dir = default_session_dirs(
                root, "haoiii", HA_LIGHTS_DIR, HA_DARKS_DIR, HA_FLATS_DIR, HA_BIASES_DIR
            )

            sii_lights_dir, sii_darks_dir, sii_flats_dir, sii_biases_dir = default_session_dirs(
                root, "siioiii", SII_LIGHTS_DIR, SII_DARKS_DIR, SII_FLATS_DIR, SII_BIASES_DIR
            )

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
        # Which sessions are active?
        # ----------------------------------------------------

        sessions = []

        if has_files(ha_lights_dir):
            sessions.append({
                "key": "ha",
                "label": "Ha",
                "lights": ha_lights_dir,
                "darks": ha_darks_dir,
                "flats": ha_flats_dir,
                "biases": ha_biases_dir,
            })

        if has_files(sii_lights_dir):
            sessions.append({
                "key": "sii",
                "label": "SII",
                "lights": sii_lights_dir,
                "darks": sii_darks_dir,
                "flats": sii_flats_dir,
                "biases": sii_biases_dir,
            })

        if not sessions:
            raise RuntimeError(
                "No light frames found. Provide a non-empty lights_haoiii/ "
                "and/or lights_siioiii/ folder."
            )

        log("==========================================")
        log("OSC narrowband preprocess")
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
            log(f"Processing {session['label']}-OIII session")
            log("------------------------------------------")
            results_by_key[session["key"]] = process_session(session)

        # ----------------------------------------------------
        # Combine OIII across every session into one sequence and
        # stack it once - this is what gives the deeper/better SNR
        # combined OIII master when both sessions are provided.
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
        # Align every present final channel (Ha, SII, combined OIII)
        # together on one shared pixel grid.
        # ----------------------------------------------------

        log("------------------------------------------")
        log("Aligning all final channel masters...")
        log("------------------------------------------")

        final_dir = process / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        channel_indices = {}
        idx = 0

        for key in ("ha", "sii"):
            if key in results_by_key:
                idx += 1
                channel_indices[key] = idx
                shutil.move(
                    str(results_by_key[key]["red_final_path"]),
                    str(final_dir / f"final_{idx:05d}.fit")
                )

        idx += 1
        oiii_idx = idx
        shutil.move(str(oiii_final_path), str(final_dir / f"final_{idx:05d}.fit"))

        cd(final_dir)

        #
        # -2pass here only computes the transform; seqapplyreg then
        # crops every final_NNNNN to their mutual common area
        # (-framing=min), which is required for them to come out the
        # same pixel size - they were cropped independently (each to
        # its own sequence's overlap) during the per-channel/combined
        # stacks above, so their sizes can differ by a few pixels
        # before this step.
        #
        # -transf=similarity (shift + rotation + uniform scale) rather
        # than a plain shift: different sessions/nights commonly have
        # slightly different field rotation, and Ha's reconstruction
        # (half-res red extraction, true-drizzled 2x) vs OIII's
        # (native-res, red interpolated away) don't have any guarantee
        # of landing on exactly scale-matched grids. A plain shift
        # can't correct either of those, leaving a small residual
        # misalignment. affine/homography would handle even more
        # distortion, but risk overfitting with the few, often noisy
        # star pairs available between narrowband channels.
        #

        siril.cmd("register", "final", "-2pass", "-transf=similarity")
        siril.cmd("seqapplyreg", "final", "-interp=lanczos4", "-framing=min")

        # ----------------------------------------------------
        # Background-only match: every non-reference channel gets
        # its median (background level) shifted to equal the
        # reference channel's median. Contrast/noise (MAD) is left
        # untouched on every channel, including the reference, so no
        # channel's noise gets amplified by this step - it only
        # gives every channel the same black point for a clean
        # composite. Reference = Ha if present, else SII.
        # ----------------------------------------------------

        reference_key = "ha" if "ha" in channel_indices else "sii"
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
