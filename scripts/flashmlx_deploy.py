#!/usr/bin/env python3
"""
FlashMLX One-Click Installer

Deploy FlashMLX KV Cache optimizations to any mlx-lm installation.

Usage:
    python3 flashmlx_deploy.py install /path/to/mlx_lm
    python3 flashmlx_deploy.py install --auto-detect /path/to/project
    python3 flashmlx_deploy.py uninstall /path/to/mlx_lm
    python3 flashmlx_deploy.py verify /path/to/mlx_lm
    python3 flashmlx_deploy.py info /path/to/mlx_lm
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import sys
import textwrap

VERSION = "0.3.1"

# FlashMLX source root (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_MODELS = os.path.join(SCRIPT_DIR, "mlx-lm-source", "mlx_lm", "models")
SOURCE_CALIBRATIONS = os.path.join(SCRIPT_DIR, "calibrations")

# 5 core files to copy into target/models/
FLASHMLX_FILES = [
    "triple_layer_cache.py",
    "cache_factory.py",
    "double_layer_cache.py",
    "quantization_strategies.py",
    "load_characteristics.py",
]

MANIFEST_NAME = ".flashmlx_manifest.json"
BACKUP_DIR_NAME = ".flashmlx_backup"


# ─── Colors ──────────────────────────────────────────────────────────

class C:
    OK = "\033[92m"
    WARN = "\033[93m"
    ERR = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"

def ok(msg):    print(f"{C.OK}[OK]{C.END} {msg}")
def warn(msg):  print(f"{C.WARN}[WARN]{C.END} {msg}")
def err(msg):   print(f"{C.ERR}[ERR]{C.END} {msg}")
def info(msg):  print(f"{C.BOLD}[INFO]{C.END} {msg}")
def step(n, msg): print(f"\n{C.BOLD}Step {n}:{C.END} {msg}")


# ─── Detection ───────────────────────────────────────────────────────

def detect_mlx_lm(path):
    """
    Auto-detect mlx_lm installation path.

    Tries in order:
    1. path itself is mlx_lm/ (has models/cache.py)
    2. path/mlx_lm/ (source checkout)
    3. path/venv/**/mlx_lm/ (virtualenv)
    4. path/**/site-packages/mlx_lm/ (any site-packages)
    """
    candidates = []

    # 1. Direct path
    if os.path.isfile(os.path.join(path, "models", "cache.py")):
        candidates.append(path)

    # 2. path/mlx_lm/
    sub = os.path.join(path, "mlx_lm")
    if os.path.isfile(os.path.join(sub, "models", "cache.py")):
        candidates.append(sub)

    # 3. venv search
    venv_patterns = [
        os.path.join(path, "venv", "lib", "python*", "site-packages", "mlx_lm"),
        os.path.join(path, ".venv", "lib", "python*", "site-packages", "mlx_lm"),
    ]
    for pattern in venv_patterns:
        for match in sorted(glob.glob(pattern)):
            if os.path.isfile(os.path.join(match, "models", "cache.py")):
                candidates.append(match)

    # 4. Broader site-packages search
    sp_pattern = os.path.join(path, "**", "site-packages", "mlx_lm")
    for match in sorted(glob.glob(sp_pattern, recursive=True)):
        if os.path.isfile(os.path.join(match, "models", "cache.py")):
            if match not in candidates:
                candidates.append(match)

    return candidates


def validate_target(target):
    """Validate that target is a valid mlx_lm directory."""
    cache_py = os.path.join(target, "models", "cache.py")
    gen_py = os.path.join(target, "generate.py")
    if not os.path.isfile(cache_py):
        err(f"Not a valid mlx_lm directory: {target}")
        err(f"  Missing: models/cache.py")
        return False
    if not os.path.isfile(gen_py):
        err(f"Not a valid mlx_lm directory: {target}")
        err(f"  Missing: generate.py")
        return False
    return True


# ─── Backup ──────────────────────────────────────────────────────────

def backup_file(target, rel_path):
    """Backup a file before modifying it."""
    src = os.path.join(target, rel_path)
    if not os.path.exists(src):
        return

    backup_dir = os.path.join(target, BACKUP_DIR_NAME)
    dst_dir = os.path.join(backup_dir, os.path.dirname(rel_path))
    os.makedirs(dst_dir, exist_ok=True)

    dst = os.path.join(backup_dir, rel_path)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


# ─── Patching ────────────────────────────────────────────────────────

def remove_flashmlx_patches(content):
    """Remove any existing FlashMLX patches from file content."""
    lines = content.split("\n")
    result = []
    skip = False
    for line in lines:
        if "# FlashMLX: BEGIN" in line:
            skip = True
            continue
        if "# FlashMLX: END" in line:
            skip = False
            continue
        if skip:
            continue
        # Remove inline FlashMLX additions (parameter lines)
        if "# FlashMLX: added" in line:
            continue
        result.append(line)
    return "\n".join(result)


def patch_cache_py(target):
    """Patch models/cache.py to add FlashMLX parameters to make_prompt_cache."""
    cache_path = os.path.join(target, "models", "cache.py")
    with open(cache_path, "r") as f:
        content = f.read()

    # Remove old patches first (idempotent)
    content = remove_flashmlx_patches(content)

    # --- Patch 1: Add parameters to make_prompt_cache signature ---
    old_sig = (
        "def make_prompt_cache(\n"
        "    model: nn.Module,\n"
        "    max_kv_size: Optional[int] = None,\n"
        ") -> List[Any]:"
    )
    new_sig = (
        "def make_prompt_cache(\n"
        "    model: nn.Module,\n"
        "    max_kv_size: Optional[int] = None,\n"
        "    kv_cache: Optional[str] = None,  # FlashMLX: added\n"
        "    kv_calibration: Optional[str] = None,  # FlashMLX: added\n"
        "    kv_compression_ratio: Optional[float] = None,  # FlashMLX: added\n"
        ") -> List[Any]:"
    )

    if old_sig not in content:
        err("Cannot patch cache.py: make_prompt_cache signature not found.")
        err("  The mlx-lm version may be incompatible. Try manual installation.")
        return False

    content = content.replace(old_sig, new_sig)

    # --- Patch 2: Add factory dispatch after make_cache check ---
    old_dispatch = "    if hasattr(model, \"make_cache\"):\n        return model.make_cache()\n"
    factory_block = textwrap.dedent("""\
    if hasattr(model, "make_cache"):
        return model.make_cache()

    # FlashMLX: BEGIN
    if kv_cache is not None:
        from mlx_lm.models.cache_factory import make_optimized_cache
        kwargs = dict(
            strategy=kv_cache,
            calibration_file=kv_calibration,
            max_kv_size=max_kv_size,
        )
        if kv_compression_ratio is not None:
            kwargs["compression_ratio"] = kv_compression_ratio
        return make_optimized_cache(model, **kwargs)
    # FlashMLX: END
""")
    # Indent the block to match
    factory_lines = ["    " + line if line.strip() else "" for line in factory_block.split("\n")]
    new_dispatch = "\n".join(factory_lines) + "\n"

    if old_dispatch not in content:
        err("Cannot patch cache.py: make_cache dispatch block not found.")
        return False

    content = content.replace(old_dispatch, new_dispatch)

    with open(cache_path, "w") as f:
        f.write(content)
    return True


def patch_generate_py(target):
    """Patch generate.py to add FlashMLX parameters to generate_step."""
    gen_path = os.path.join(target, "generate.py")
    with open(gen_path, "r") as f:
        content = f.read()

    # Remove old patches first (idempotent)
    content = remove_flashmlx_patches(content)

    # --- Patch 1: Add parameters to generate_step signature ---
    old_param = "    input_embeddings: Optional[mx.array] = None,\n) -> Generator[Tuple[mx.array, mx.array], None, None]:"
    new_param = (
        "    input_embeddings: Optional[mx.array] = None,\n"
        "    kv_cache: Optional[str] = None,  # FlashMLX: added\n"
        "    kv_calibration: Optional[str] = None,  # FlashMLX: added\n"
        "    kv_compression_ratio: Optional[float] = None,  # FlashMLX: added\n"
        ") -> Generator[Tuple[mx.array, mx.array], None, None]:"
    )

    if old_param not in content:
        err("Cannot patch generate.py: generate_step signature not found.")
        err("  The mlx-lm version may be incompatible. Try manual installation.")
        return False

    content = content.replace(old_param, new_param)

    # --- Patch 2: Update make_prompt_cache call to pass new params ---
    old_call = (
        "        prompt_cache = cache.make_prompt_cache(\n"
        "            model,\n"
        "            max_kv_size=max_kv_size,\n"
        "        )"
    )
    new_call = (
        "        prompt_cache = cache.make_prompt_cache(\n"
        "            model,\n"
        "            max_kv_size=max_kv_size,\n"
        "            kv_cache=kv_cache,  # FlashMLX: added\n"
        "            kv_calibration=kv_calibration,  # FlashMLX: added\n"
        "            kv_compression_ratio=kv_compression_ratio,  # FlashMLX: added\n"
        "        )"
    )

    if old_call not in content:
        err("Cannot patch generate.py: make_prompt_cache call not found.")
        return False

    content = content.replace(old_call, new_call)

    with open(gen_path, "w") as f:
        f.write(content)
    return True


# ─── Install ─────────────────────────────────────────────────────────

def cmd_install(target, force=False, no_calibration=False, auto_detect=False):
    """Install FlashMLX to target mlx_lm directory."""
    print(f"\n{C.BOLD}FlashMLX Installer v{VERSION}{C.END}")
    print(f"{'=' * 50}")

    # Auto-detect if requested
    if auto_detect:
        info(f"Auto-detecting mlx_lm in: {target}")
        candidates = detect_mlx_lm(target)
        if not candidates:
            err(f"No mlx_lm installation found in: {target}")
            err("  Searched: direct path, mlx_lm/, venv/, site-packages/")
            return False
        if len(candidates) == 1:
            target = candidates[0]
            ok(f"Found mlx_lm at: {target}")
        else:
            info("Multiple mlx_lm installations found:")
            for i, c in enumerate(candidates):
                print(f"  [{i+1}] {c}")
            try:
                choice = input(f"\nSelect [1-{len(candidates)}]: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(candidates):
                    target = candidates[idx]
                else:
                    err("Invalid selection.")
                    return False
            except (ValueError, EOFError):
                err("Invalid selection.")
                return False

    # Validate
    if not validate_target(target):
        return False

    # Check existing installation
    manifest_path = os.path.join(target, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not force:
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing_ver = manifest.get("version", "unknown")
        if existing_ver == VERSION:
            warn(f"FlashMLX v{VERSION} already installed. Use --force to reinstall.")
            return True
        info(f"Upgrading FlashMLX v{existing_ver} -> v{VERSION}")

    # Validate source files exist
    for fname in FLASHMLX_FILES:
        src = os.path.join(SOURCE_MODELS, fname)
        if not os.path.exists(src):
            err(f"Source file not found: {src}")
            err("  FlashMLX source tree may be incomplete.")
            return False

    # ── Step 1: Backup ──
    step(1, "Backing up original files")
    files_to_backup = [
        os.path.join("models", "cache.py"),
        "generate.py",
    ]
    for rel in files_to_backup:
        backup_file(target, rel)
        print(f"  {C.DIM}Backed up: {rel}{C.END}")

    backup_dir = os.path.join(target, BACKUP_DIR_NAME)
    ok(f"Backups saved to: {backup_dir}")

    # ── Step 2: Copy core files ──
    step(2, "Copying FlashMLX core files")
    models_dir = os.path.join(target, "models")
    copied_files = []
    for fname in FLASHMLX_FILES:
        src = os.path.join(SOURCE_MODELS, fname)
        dst = os.path.join(models_dir, fname)
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"  {C.DIM}{fname} ({size_kb:.0f} KB){C.END}")
        copied_files.append(os.path.join("models", fname))
    ok(f"Copied {len(FLASHMLX_FILES)} files to {models_dir}")

    # ── Step 3: Patch cache.py ──
    step(3, "Patching models/cache.py")
    if not patch_cache_py(target):
        err("Failed to patch cache.py. Rolling back...")
        cmd_uninstall(target)
        return False
    ok("Added kv_cache/kv_calibration/kv_compression_ratio to make_prompt_cache")

    # ── Step 4: Patch generate.py ──
    step(4, "Patching generate.py")
    if not patch_generate_py(target):
        err("Failed to patch generate.py. Rolling back...")
        cmd_uninstall(target)
        return False
    ok("Added kv_cache/kv_calibration/kv_compression_ratio to generate_step")

    # ── Step 5: Copy calibrations ──
    step(5, "Copying calibration files")
    calib_dst = os.path.join(os.path.dirname(target), "flashmlx_calibrations")
    os.makedirs(calib_dst, exist_ok=True)
    calib_files = sorted(glob.glob(os.path.join(SOURCE_CALIBRATIONS, "*.pkl")))

    if calib_files:
        for src in calib_files:
            fname = os.path.basename(src)
            dst = os.path.join(calib_dst, fname)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"  {C.DIM}{fname} ({size_mb:.1f} MB){C.END}")
        ok(f"Copied {len(calib_files)} calibration files to {calib_dst}")
    else:
        warn("No calibration files found. AM compression will not be available.")
        warn(f"  Expected in: {SOURCE_CALIBRATIONS}")

    # ── Step 6: Write manifest ──
    step(6, "Writing installation manifest")
    manifest = {
        "version": VERSION,
        "installed_at": datetime.datetime.now().isoformat(),
        "source": SCRIPT_DIR,
        "files_copied": copied_files,
        "files_patched": ["models/cache.py", "generate.py"],
        "backup_dir": BACKUP_DIR_NAME,
        "calibrations_dir": calib_dst,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    ok(f"Manifest written: {manifest_path}")

    # ── Step 7: Verify ──
    step(7, "Verifying installation")
    if cmd_verify(target, quiet=True):
        ok("Verification passed")
    else:
        warn("Verification had issues (see above). Installation may be incomplete.")

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print(f"{C.OK}{C.BOLD}FlashMLX v{VERSION} installed successfully!{C.END}")
    print(f"\nTarget: {target}")
    print(f"Calibrations: {calib_dst}")
    print(f"\n{C.BOLD}Quick Start:{C.END}")
    print(f"  from mlx_lm import load, generate")
    print(f"  model, tokenizer = load('model_path')")
    print(f"")
    print(f"  # Triple (Q4_0 only, lossless, ~40% prefill savings)")
    print(f"  text = generate(model, tokenizer, prompt, kv_cache='triple')")
    print(f"")
    print(f"  # Triple + AM (50%+ memory savings, faster TG)")
    print(f"  text = generate(model, tokenizer, prompt,")
    print(f"      kv_cache='triple_am',")
    print(f"      kv_calibration='{calib_dst}/<model_calibration>.pkl')")
    print(f"")
    print(f"  # Adaptive ratio (auto-selects based on context length)")
    print(f"  text = generate(model, tokenizer, prompt,")
    print(f"      kv_cache='triple_am',")
    print(f"      kv_calibration='{calib_dst}/<model_calibration>.pkl',")
    print(f"      kv_compression_ratio=0)")

    # ── Calibration prompt ──
    if not no_calibration:
        print(f"\n{'=' * 50}")
        prompt_calibration(calib_dst)

    return True


# ─── Calibration ─────────────────────────────────────────────────────

def prompt_calibration(calib_dir):
    """Ask user if they want to generate model-specific calibration."""
    info("Calibration files available:")
    calib_files = sorted(glob.glob(os.path.join(calib_dir, "*.pkl")))
    if calib_files:
        for f in calib_files:
            fname = os.path.basename(f)
            size_mb = os.path.getsize(f) / (1024 * 1024)
            # Extract model name from filename
            model_name = fname.replace("am_calibration_", "").replace("_onpolicy", "").replace(".pkl", "")
            print(f"  {fname} ({size_mb:.1f} MB) - for {model_name}")
    else:
        print(f"  (none)")

    print(f"\n{C.BOLD}Want to generate calibration for a different model?{C.END}")
    print(f"  Calibration enables AM compression (50%+ memory savings, faster TG).")
    print(f"  It requires ~5-10 minutes of GPU time per model.")

    try:
        answer = input(f"\n  Generate calibration? [y/N]: ").strip().lower()
    except EOFError:
        answer = "n"

    if answer in ("y", "yes"):
        print(f"\n{C.BOLD}To generate calibration, run:{C.END}")
        print(f"")
        print(f"  cd {SCRIPT_DIR}")
        print(f"  python3 -c \"")
        print(f"  import sys; sys.path.insert(0, 'mlx-lm-source')")
        print(f"  from mlx_lm import load")
        print(f"  from mlx_lm.models.double_layer_cache import CalibrationRegistry")
        print(f"  model, tokenizer = load('/path/to/your/model')")
        print(f"  reg = CalibrationRegistry()")
        print(f"  reg.calibrate_model(model, tokenizer, output_dir='{calib_dir}')\"")
        print(f"")
        print(f"  Replace '/path/to/your/model' with your model's path.")
    else:
        ok("Skipping calibration. You can generate it later.")


# ─── Uninstall ───────────────────────────────────────────────────────

def cmd_uninstall(target):
    """Restore original files from backup."""
    print(f"\n{C.BOLD}FlashMLX Uninstaller{C.END}")
    print(f"{'=' * 50}")

    if not validate_target(target):
        # Still try to restore even if validation fails
        pass

    backup_dir = os.path.join(target, BACKUP_DIR_NAME)
    manifest_path = os.path.join(target, MANIFEST_NAME)

    if not os.path.exists(backup_dir):
        err(f"No backup directory found: {backup_dir}")
        err("  Cannot restore original files. Manual restoration needed.")
        return False

    # Restore backed up files
    info("Restoring original files from backup")
    restored = 0
    for root, dirs, files in os.walk(backup_dir):
        for fname in files:
            backup_path = os.path.join(root, fname)
            rel_path = os.path.relpath(backup_path, backup_dir)
            target_path = os.path.join(target, rel_path)
            shutil.copy2(backup_path, target_path)
            print(f"  Restored: {rel_path}")
            restored += 1

    # Remove FlashMLX files
    info("Removing FlashMLX files")
    for fname in FLASHMLX_FILES:
        fpath = os.path.join(target, "models", fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  Removed: models/{fname}")

    # Remove manifest
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    # Remove backup dir
    shutil.rmtree(backup_dir, ignore_errors=True)

    ok(f"Restored {restored} files. FlashMLX removed from {target}")
    return True


# ─── Verify ──────────────────────────────────────────────────────────

def cmd_verify(target, quiet=False):
    """Verify FlashMLX installation."""
    if not quiet:
        print(f"\n{C.BOLD}FlashMLX Verification{C.END}")
        print(f"{'=' * 50}")

    all_ok = True

    # 1. Check core files exist
    for fname in FLASHMLX_FILES:
        fpath = os.path.join(target, "models", fname)
        if os.path.exists(fpath):
            if not quiet:
                ok(f"  {fname} exists")
        else:
            err(f"  {fname} MISSING")
            all_ok = False

    # 2. Check patches
    cache_path = os.path.join(target, "models", "cache.py")
    with open(cache_path) as f:
        cache_content = f.read()

    if "kv_cache: Optional[str]" in cache_content:
        if not quiet:
            ok("  cache.py patched (kv_cache parameter)")
    else:
        err("  cache.py NOT patched (missing kv_cache parameter)")
        all_ok = False

    if "cache_factory" in cache_content:
        if not quiet:
            ok("  cache.py patched (factory dispatch)")
    else:
        err("  cache.py NOT patched (missing factory dispatch)")
        all_ok = False

    gen_path = os.path.join(target, "generate.py")
    with open(gen_path) as f:
        gen_content = f.read()

    if "kv_cache: Optional[str]" in gen_content:
        if not quiet:
            ok("  generate.py patched (kv_cache parameter)")
    else:
        err("  generate.py NOT patched (missing kv_cache parameter)")
        all_ok = False

    if "kv_calibration=kv_calibration" in gen_content:
        if not quiet:
            ok("  generate.py patched (kv_calibration passthrough)")
    else:
        err("  generate.py NOT patched (missing kv_calibration passthrough)")
        all_ok = False

    # 3. Check calibration files
    manifest_path = os.path.join(target, MANIFEST_NAME)
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        calib_dir = manifest.get("calibrations_dir", "")
        if calib_dir and os.path.exists(calib_dir):
            n_calib = len(glob.glob(os.path.join(calib_dir, "*.pkl")))
            if not quiet:
                ok(f"  {n_calib} calibration file(s) at {calib_dir}")
        else:
            if not quiet:
                warn(f"  Calibration directory not found: {calib_dir}")
    else:
        if not quiet:
            warn("  No manifest found")

    # 4. Import test (try importing in subprocess to avoid polluting our namespace)
    import subprocess
    test_code = (
        f"import sys; sys.path.insert(0, '{os.path.dirname(target)}'); "
        f"from mlx_lm.models.triple_layer_cache import TripleLayerKVCache; "
        f"from mlx_lm.models.cache_factory import make_optimized_cache; "
        f"print('OK')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            if not quiet:
                ok("  Import test passed (TripleLayerKVCache, cache_factory)")
        else:
            stderr = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown error"
            err(f"  Import test FAILED: {stderr}")
            all_ok = False
    except subprocess.TimeoutExpired:
        warn("  Import test timed out (30s)")
    except FileNotFoundError:
        warn("  Cannot run import test (python3 not found)")

    if not quiet:
        print()
        if all_ok:
            ok("All checks passed!")
        else:
            err("Some checks failed. See above.")

    return all_ok


# ─── Info ────────────────────────────────────────────────────────────

def cmd_info(target):
    """Show FlashMLX installation info."""
    print(f"\n{C.BOLD}FlashMLX Installation Info{C.END}")
    print(f"{'=' * 50}")

    manifest_path = os.path.join(target, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        err(f"FlashMLX not installed at: {target}")
        err(f"  No manifest found: {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"  Version:       {manifest.get('version', 'unknown')}")
    print(f"  Installed at:  {manifest.get('installed_at', 'unknown')}")
    print(f"  Source:        {manifest.get('source', 'unknown')}")
    print(f"  Target:        {target}")
    print(f"  Backup:        {os.path.join(target, manifest.get('backup_dir', ''))}")
    print(f"  Calibrations:  {manifest.get('calibrations_dir', 'none')}")
    print(f"\n  Files copied:")
    for f in manifest.get("files_copied", []):
        print(f"    {f}")
    print(f"\n  Files patched:")
    for f in manifest.get("files_patched", []):
        print(f"    {f}")

    # Show available calibrations
    calib_dir = manifest.get("calibrations_dir", "")
    if calib_dir and os.path.exists(calib_dir):
        calib_files = sorted(glob.glob(os.path.join(calib_dir, "*.pkl")))
        if calib_files:
            print(f"\n  Calibration files:")
            for cf in calib_files:
                size_mb = os.path.getsize(cf) / (1024 * 1024)
                print(f"    {os.path.basename(cf)} ({size_mb:.1f} MB)")

    return True


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FlashMLX One-Click Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Install to a specific mlx_lm directory
              python3 flashmlx_deploy.py install /path/to/mlx_lm

              # Auto-detect mlx_lm in a project (searches venv, site-packages)
              python3 flashmlx_deploy.py install --auto-detect /path/to/project

              # Uninstall (restore original files from backup)
              python3 flashmlx_deploy.py uninstall /path/to/mlx_lm

              # Verify installation
              python3 flashmlx_deploy.py verify /path/to/mlx_lm
        """),
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # install
    p_install = subparsers.add_parser("install", help="Install FlashMLX")
    p_install.add_argument("target", help="Target mlx_lm directory or project path")
    p_install.add_argument("--auto-detect", action="store_true",
                          help="Auto-detect mlx_lm location in project")
    p_install.add_argument("--force", action="store_true",
                          help="Force reinstall even if same version")
    p_install.add_argument("--no-calibration", action="store_true",
                          help="Skip calibration prompt")

    # uninstall
    p_uninstall = subparsers.add_parser("uninstall", help="Uninstall FlashMLX")
    p_uninstall.add_argument("target", help="Target mlx_lm directory")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify FlashMLX installation")
    p_verify.add_argument("target", help="Target mlx_lm directory")

    # info
    p_info = subparsers.add_parser("info", help="Show installation info")
    p_info.add_argument("target", help="Target mlx_lm directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    target = os.path.abspath(args.target)

    if args.command == "install":
        success = cmd_install(
            target,
            force=args.force,
            no_calibration=args.no_calibration,
            auto_detect=args.auto_detect,
        )
    elif args.command == "uninstall":
        success = cmd_uninstall(target)
    elif args.command == "verify":
        success = cmd_verify(target)
    elif args.command == "info":
        success = cmd_info(target)
    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
