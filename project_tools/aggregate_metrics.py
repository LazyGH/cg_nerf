import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("metrics", nargs="+", help="metrics.json files produced by the evaluation scripts")
    parser.add_argument("--out_csv", default="", help="Optional output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for metric_path in args.metrics:
        with open(metric_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row = {
            "method": payload.get("method"),
            "scene": payload.get("scene"),
            "source": payload.get("source"),
            "expname": payload.get("expname"),
            "train_time_minutes": payload.get("train_time_minutes"),
            "psnr": payload.get("avg", {}).get("psnr"),
            "ssim": payload.get("avg", {}).get("ssim"),
            "lpips_alex": payload.get("avg", {}).get("lpips_alex"),
            "lpips_vgg": payload.get("avg", {}).get("lpips_vgg"),
            "metrics_path": os.path.abspath(metric_path),
            "checkpoint": payload.get("checkpoint"),
        }
        rows.append(row)

    fieldnames = list(rows[0].keys())
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
