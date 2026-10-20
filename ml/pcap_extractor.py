"""
ml/pcap_extractor.py
--------------------
PCAP Feature Extractor for AutoMLOps.

Converts raw .pcap / .pcapng files into a pandas DataFrame of tabular network-flow
features that can be used by the standard train.py / predict.py pipeline.

Supported feature groups (all numeric):
  - Duration, packet counts, byte counts, inter-arrival times
  - Protocol distribution (TCP / UDP / ICMP / Other)
  - TCP flag counts (SYN, ACK, FIN, RST, PSH, URG)
  - Payload size stats (min, max, mean, std)
  - Unique source/destination IP and port counts per flow

Usage (standalone CLI):
    python -m ml.pcap_extractor --input data/traffic.pcap --output data/flows.csv

Usage (from mlops.yaml via train.py when dataset_path ends in .pcap/.pcapng or
is a directory of pcap files):
    dataset_path: "data/traffic.pcap"
    dataset_type: "pcap"          # or "pcap_dir" for a folder
    pcap_label_map:               # only needed for pcap_dir labelling
      "benign": 0
      "attack": 1

Required:
    pip install scapy
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("automlops.pcap_extractor")

# ── Feature columns produced by this extractor (all numeric) ─────────────────
PCAP_FEATURE_COLUMNS: List[str] = [
    "duration_sec",
    "total_packets",
    "total_bytes",
    "mean_pkt_size",
    "std_pkt_size",
    "min_pkt_size",
    "max_pkt_size",
    "mean_iat_sec",
    "std_iat_sec",
    "min_iat_sec",
    "max_iat_sec",
    "pct_tcp",
    "pct_udp",
    "pct_icmp",
    "pct_other_proto",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
]


def _extract_flows(pcap_path: str) -> pd.DataFrame:
    """
    Reads a PCAP file with scapy, groups into 5-tuple flows
    (src_ip, dst_ip, src_port, dst_port, proto), and computes
    per-flow feature vectors.

    Returns:
        DataFrame with PCAP_FEATURE_COLUMNS — one row per flow.
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP  # noqa: F401
    except ImportError:
        raise ImportError(
            "scapy is required for PCAP support.\n"
            "Install it:  pip install scapy"
        )

    logger.info(f"Reading PCAP: {pcap_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        packets = rdpcap(pcap_path)

    flows: Dict[Tuple, List[Dict[str, Any]]] = {}

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        proto = ip.proto
        src_port = dst_port = 0
        tcp_flags = 0

        if pkt.haslayer(TCP):
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            tcp_flags = int(pkt[TCP].flags)
        elif pkt.haslayer(UDP):
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        key = (src_ip, dst_ip, src_port, dst_port, proto)
        flows.setdefault(key, []).append({
            "len": len(pkt),
            "time": float(pkt.time),
            "proto": proto,
            "flags": tcp_flags,
        })

    if not flows:
        logger.warning("No IP packets found in PCAP. Returning empty DataFrame.")
        return pd.DataFrame(columns=PCAP_FEATURE_COLUMNS)

    rows = []
    for (src_ip, dst_ip, src_port, dst_port, proto), pkts in flows.items():
        times = sorted(p["time"] for p in pkts)
        lengths = [p["len"] for p in pkts]
        n = len(pkts)
        duration = times[-1] - times[0] if n > 1 else 0.0
        iats = [times[i + 1] - times[i] for i in range(len(times) - 1)] or [0.0]

        proto_vals = [p["proto"] for p in pkts]
        flags_list = [p["flags"] for p in pkts]

        rows.append({
            "duration_sec":    round(duration, 6),
            "total_packets":   n,
            "total_bytes":     sum(lengths),
            "mean_pkt_size":   round(float(np.mean(lengths)), 4),
            "std_pkt_size":    round(float(np.std(lengths)), 4),
            "min_pkt_size":    float(np.min(lengths)),
            "max_pkt_size":    float(np.max(lengths)),
            "mean_iat_sec":    round(float(np.mean(iats)), 6),
            "std_iat_sec":     round(float(np.std(iats)), 6),
            "min_iat_sec":     round(float(np.min(iats)), 6),
            "max_iat_sec":     round(float(np.max(iats)), 6),
            "pct_tcp":         round(sum(1 for x in proto_vals if x == 6) / n, 4),
            "pct_udp":         round(sum(1 for x in proto_vals if x == 17) / n, 4),
            "pct_icmp":        round(sum(1 for x in proto_vals if x == 1) / n, 4),
            "pct_other_proto": round(sum(1 for x in proto_vals if x not in (1, 6, 17)) / n, 4),
            "syn_count":       sum(1 for f in flags_list if f & 0x02),
            "ack_count":       sum(1 for f in flags_list if f & 0x10),
            "fin_count":       sum(1 for f in flags_list if f & 0x01),
            "rst_count":       sum(1 for f in flags_list if f & 0x04),
            "psh_count":       sum(1 for f in flags_list if f & 0x08),
            "urg_count":       sum(1 for f in flags_list if f & 0x20),
            "unique_src_ips":  1,     # each flow key has exactly 1 src IP
            "unique_dst_ips":  1,
            "unique_dst_ports": 1,
        })

    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} flows from {len(packets)} packets  [{pcap_path}]")
    return df


def load_pcap_as_dataframe(
    pcap_path: str,
    label_column: Optional[str] = None,
    label_value: Optional[Any] = None,
) -> pd.DataFrame:
    """
    High-level helper: parse a single PCAP into a feature DataFrame.

    Args:
        pcap_path:    Path to the .pcap or .pcapng file.
        label_column: Name of the target column to append (e.g. "label").
        label_value:  Value for every row (e.g. 0 = benign, 1 = attack).

    Returns:
        DataFrame with PCAP_FEATURE_COLUMNS (+label_column if provided).
    """
    df = _extract_flows(pcap_path)
    if label_column and label_value is not None and not df.empty:
        df[label_column] = label_value
    return df


def load_pcap_directory(
    directory: str,
    label_map: Optional[Dict[str, Any]] = None,
    label_column: str = "label",
) -> pd.DataFrame:
    """
    Batch-load a directory of PCAP files into one combined DataFrame.

    Label assignment: if label_map is provided, any file whose path
    contains a pattern key gets the corresponding label value.

    Example label_map:
        {"benign": 0, "attack": 1}
        — a file "data/pcaps/attack_capture.pcap" would get label 1.

    Args:
        directory:    Path to directory (searched recursively).
        label_map:    Optional pattern→label dict.
        label_column: Column name for the label.

    Returns:
        Concatenated DataFrame of all flows.
    """
    pcap_dir = Path(directory)
    pcap_files = list(pcap_dir.glob("**/*.pcap")) + list(pcap_dir.glob("**/*.pcapng"))

    if not pcap_files:
        logger.warning(f"No PCAP files found in {directory}")
        return pd.DataFrame(columns=PCAP_FEATURE_COLUMNS)

    frames = []
    for pcap_file in sorted(pcap_files):
        lval = None
        if label_map:
            for pattern, v in label_map.items():
                if pattern.lower() in str(pcap_file).lower():
                    lval = v
                    break

        df = load_pcap_as_dataframe(
            str(pcap_file),
            label_column=label_column if lval is not None else None,
            label_value=lval,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=PCAP_FEATURE_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(result)} flows from {len(pcap_files)} PCAP files in {directory}")
    return result


def is_pcap_dataset(dataset_path: str) -> bool:
    """Returns True when the path is a .pcap/.pcapng file or a dir that contains them."""
    p = Path(dataset_path)
    if p.suffix.lower() in (".pcap", ".pcapng"):
        return True
    if p.is_dir():
        return bool(list(p.glob("**/*.pcap")) or list(p.glob("**/*.pcapng")))
    return False


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    ap = argparse.ArgumentParser(description="Extract flow features from PCAP files")
    ap.add_argument("--input",  required=True, help=".pcap file or directory of .pcap files")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--label-column", default=None, help="Optional label column name")
    ap.add_argument("--label-value",  default=None, help="Optional label value (for single file)")
    args = ap.parse_args()

    p = Path(args.input)
    if p.is_dir():
        result_df = load_pcap_directory(str(p), label_column=args.label_column or "label")
    else:
        result_df = load_pcap_as_dataframe(
            str(p),
            label_column=args.label_column,
            label_value=args.label_value,
        )

    result_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(result_df)} flows → {args.output}")
    print(result_df.dtypes.to_string())
    print(result_df.head(3).to_string())
