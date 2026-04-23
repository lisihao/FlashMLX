#!/usr/bin/env python3
"""
Parse xctrace Metal System Trace export to extract GPU gap data.

Usage:
  # Capture a trace
  python parse_metal_trace.py capture --cmd "python my_bench.py" --duration 10

  # Parse an existing trace
  python parse_metal_trace.py parse --trace profiling_data/my_trace.gputrace

  # Capture + parse in one shot
  python parse_metal_trace.py auto --cmd "python my_bench.py"
"""

import argparse
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional


def capture_trace(
    cmd: List[str],
    output_path: str,
    duration_ms: int = 10000,
) -> str:
    """Run a command under xctrace Metal System Trace.

    Args:
        cmd: Command and arguments to launch
        output_path: Path for .gputrace output
        duration_ms: Maximum recording time in milliseconds

    Returns:
        Path to the .gputrace file
    """
    trace_path = Path(output_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing trace if present (xctrace won't overwrite)
    if trace_path.exists():
        subprocess.run(["rm", "-rf", str(trace_path)], check=True)

    xctrace_cmd = [
        "xcrun", "xctrace", "record",
        "--template", "Metal System Trace",
        "--output", str(trace_path),
        "--time-limit", f"{duration_ms}ms",
        "--launch", "--",
    ] + cmd

    print(f"Capturing Metal trace ({duration_ms}ms max)...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(xctrace_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"xctrace failed (rc={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        return ""

    if not trace_path.exists():
        print("Trace file not created")
        return ""

    print(f"Trace captured: {trace_path}")
    return str(trace_path)


def export_trace(trace_path: str, output_dir: Optional[str] = None) -> str:
    """Export a .gputrace to XML for parsing.

    Args:
        trace_path: Path to .gputrace file
        output_dir: Directory for XML export (auto-created if None)

    Returns:
        Path to the export directory
    """
    if output_dir is None:
        output_dir = str(Path(trace_path).with_suffix(".export"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "xcrun", "xctrace", "export",
        "--input", trace_path,
        "--output", output_dir,
    ]

    print(f"Exporting trace to XML...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"xctrace export failed (rc={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        return ""

    print(f"Export dir: {output_dir}")
    return output_dir


def parse_gpu_timeline(export_dir: str) -> dict:
    """Parse exported trace XML to extract GPU timeline and gaps.

    Looks for GPU command buffer start/end events to compute:
    - Total time, active time, idle time
    - Gap durations between command buffers
    - Gap ratio (idle / total)

    Returns:
        Dictionary with gap analysis results
    """
    export_path = Path(export_dir)

    # Find XML files in export
    xml_files = list(export_path.glob("**/*.xml"))
    if not xml_files:
        return {
            "status": "no_xml",
            "error": f"No XML files found in {export_dir}",
        }

    gpu_events = []

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError as e:
            continue

        # Walk all elements looking for GPU-related timing data
        # xctrace exports vary by version; we look for common patterns
        for elem in root.iter():
            tag = elem.tag.lower() if elem.tag else ""
            attribs = elem.attrib

            # Look for GPU command buffer events
            if "gpu" in tag or "command-buffer" in tag or "metal" in tag:
                start = attribs.get("start-time") or attribs.get("timestamp")
                duration = attribs.get("duration")
                if start is not None:
                    try:
                        start_ns = _parse_time(start)
                        dur_ns = _parse_time(duration) if duration else 0
                        gpu_events.append({
                            "start_ns": start_ns,
                            "end_ns": start_ns + dur_ns,
                            "duration_ns": dur_ns,
                        })
                    except (ValueError, TypeError):
                        continue

            # Also check for row-based data (common in xctrace exports)
            if tag == "row":
                for child in elem:
                    text = (child.text or "").lower()
                    if "gpu" in text or "command buffer" in text:
                        # Try to extract timing from sibling columns
                        cols = list(elem)
                        if len(cols) >= 3:
                            try:
                                start_ns = _parse_time(cols[0].text)
                                dur_ns = _parse_time(cols[1].text)
                                gpu_events.append({
                                    "start_ns": start_ns,
                                    "end_ns": start_ns + dur_ns,
                                    "duration_ns": dur_ns,
                                })
                            except (ValueError, TypeError):
                                pass
                        break

    if not gpu_events:
        return {
            "status": "no_gpu_events",
            "note": "Could not find GPU command buffer events in trace export. "
                    "Try opening the .gputrace in Instruments.app for manual analysis.",
            "xml_files_checked": len(xml_files),
        }

    # Sort by start time
    gpu_events.sort(key=lambda e: e["start_ns"])

    # Compute gaps
    gaps = []
    for i in range(1, len(gpu_events)):
        gap_ns = gpu_events[i]["start_ns"] - gpu_events[i - 1]["end_ns"]
        if gap_ns > 0:
            gaps.append(gap_ns)

    total_span_ns = gpu_events[-1]["end_ns"] - gpu_events[0]["start_ns"]
    total_active_ns = sum(e["duration_ns"] for e in gpu_events)
    total_gap_ns = sum(gaps)

    total_span_ms = total_span_ns / 1e6
    total_gap_ms = total_gap_ns / 1e6
    gap_ratio = total_gap_ns / total_span_ns if total_span_ns > 0 else 0

    return {
        "status": "ok",
        "num_command_buffers": len(gpu_events),
        "total_span_ms": total_span_ms,
        "total_active_ms": total_active_ns / 1e6,
        "total_gap_ms": total_gap_ms,
        "gap_ratio": gap_ratio,
        "num_gaps": len(gaps),
        "mean_gap_ms": (total_gap_ms / len(gaps)) if gaps else 0,
        "max_gap_ms": (max(gaps) / 1e6) if gaps else 0,
        "min_gap_ms": (min(gaps) / 1e6) if gaps else 0,
    }


def _parse_time(value) -> int:
    """Parse a time value to nanoseconds. Handles ns, us, ms, s suffixes."""
    if value is None:
        raise ValueError("None value")

    s = str(value).strip()

    # Pure numeric — assume nanoseconds
    try:
        return int(float(s))
    except ValueError:
        pass

    # With suffix
    for suffix, multiplier in [("ns", 1), ("us", 1000), ("ms", 1_000_000), ("s", 1_000_000_000)]:
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            return int(float(num) * multiplier)

    raise ValueError(f"Cannot parse time: {value}")


def analyze_from_trace(trace_path: str) -> dict:
    """Export + parse a .gputrace file in one call."""
    export_dir = export_trace(trace_path)
    if not export_dir:
        return {"status": "export_failed"}
    return parse_gpu_timeline(export_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Metal System Trace parser")
    sub = parser.add_subparsers(dest="action")

    cap = sub.add_parser("capture", help="Capture a Metal trace")
    cap.add_argument("--cmd", required=True, help="Command to run")
    cap.add_argument("--output", default="profiling_data/trace.gputrace")
    cap.add_argument("--duration", type=int, default=10000, help="Max duration ms")

    par = sub.add_parser("parse", help="Parse an existing trace")
    par.add_argument("--trace", required=True, help="Path to .gputrace")

    auto = sub.add_parser("auto", help="Capture + parse")
    auto.add_argument("--cmd", required=True)
    auto.add_argument("--output", default="profiling_data/trace.gputrace")
    auto.add_argument("--duration", type=int, default=10000)

    args = parser.parse_args()

    if args.action == "capture":
        capture_trace(args.cmd.split(), args.output, args.duration)

    elif args.action == "parse":
        result = analyze_from_trace(args.trace)
        print(json.dumps(result, indent=2))

    elif args.action == "auto":
        trace_path = capture_trace(args.cmd.split(), args.output, args.duration)
        if trace_path:
            result = analyze_from_trace(trace_path)
            print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
