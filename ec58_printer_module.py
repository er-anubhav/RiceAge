import os
import time
from datetime import datetime

def format_and_print_ec58(data: dict, print_to_printer: bool = True) -> dict:
    """
    Formats the rice grain analysis report for Everycom EC58 thermal printer (58mm, 32 chars/line).
    If print_to_printer is True, attempts to write directly to /dev/usb/lp0 (or fallback /dev/usb/lp1).
    """
    total_grains = data.get("total_grains", data.get("total_objects", 0))
    batch_no = data.get("batch_no", "N/A")
    disposition = data.get("disposition", "SUBSTANDARD")
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")

    avg_l = float(data.get("avg_length_mm", 0.0))
    avg_b = float(data.get("avg_breadth_mm", 0.0))
    lb = float(data.get("length_breadth_ratio", 0.0))
    if lb <= 0 and avg_b > 0:
        lb = avg_l / avg_b
    grade = "Long Slender" if lb >= 3.0 else ("Medium Slender" if lb >= 2.5 else "Short Bold")

    # Chemical Age
    ra = data.get("rice_age", {})
    age_str = ra.get("dominant_age_bracket", ra.get("primary_age", "N/A"))

    stats = data.get("stats", [])

    # Plain text preview (32 columns per line for 58mm printer)
    DIVIDER = "--------------------------------"
    DOUBLE_DIVIDER = "================================"

    receipt_lines = []
    receipt_lines.append(DOUBLE_DIVIDER)
    receipt_lines.append("   I.T.S GRAIN AGE & QUALITY    ")
    receipt_lines.append("        ANALYZER REPORT         ")
    receipt_lines.append(DOUBLE_DIVIDER)
    receipt_lines.append(f"Date/Time: {now_str}")
    receipt_lines.append(f"Batch No : {batch_no}")
    receipt_lines.append(f"Total Grn: {total_grains}")
    receipt_lines.append(DIVIDER)
    receipt_lines.append("PARAM            CNT   %   LIMIT")
    receipt_lines.append(DIVIDER)

    for s in stats:
        name = s.get("param", "")
        short_names = {
            "Good / Head Rice": "Head Rice",
            "Big Brokens (25-75%)": "Big Brokens",
            "Small Brokens (<25%)": "Small Brokens",
            "Foreign Matter": "Foreign Mat",
            "Damaged Grains": "Damaged",
            "Discoloured Grains": "Discoloured",
            "Chalky Grains": "Chalky",
            "Unresolved Clusters / Review Pending": "Clusters"
        }
        p_name = short_names.get(name, name[:14]).ljust(14)[:14]
        cnt = str(s.get("count", 0)).rjust(4)[:4]
        pct_val = s.get("mass_pct", s.get("pct", 0.0))
        pct_str = f"{pct_val:.1f}%".rjust(5)[:5]
        
        limit_str = s.get("limit", "-")
        lim_clean = limit_str.replace("Min ", ">=").replace("Max ", "<=").replace(".00%", "%").replace(" (Informational)", "")[:6].rjust(6)
        
        receipt_lines.append(f"{p_name} {cnt} {pct_str} {lim_clean}")

    receipt_lines.append(DIVIDER)
    receipt_lines.append("KERNEL GEOMETRY & AGE METRICS")
    receipt_lines.append(DIVIDER)
    receipt_lines.append(f"Avg Length  : {avg_l:.3f} mm")
    receipt_lines.append(f"Avg Breadth : {avg_b:.3f} mm")
    receipt_lines.append(f"L/B Ratio   : {lb:.2f} ({grade})")
    receipt_lines.append(f"Chemical Age: {age_str}")
    receipt_lines.append(DOUBLE_DIVIDER)
    receipt_lines.append(f"VERDICT: {disposition.upper()}")
    receipt_lines.append(DOUBLE_DIVIDER)
    receipt_lines.append("  * Graded per AP Civil Supplies")
    receipt_lines.append("     & Codex CXS 198 Standards  ")
    receipt_lines.append(DOUBLE_DIVIDER)

    text_content = "\n".join(receipt_lines)

    status_result = {
        "success": False,
        "preview": text_content,
        "device": None,
        "message": ""
    }

    if not print_to_printer:
        status_result["success"] = True
        status_result["message"] = "Preview generated successfully"
        return status_result

    # Connect to Everycom EC58 printer
    devices = ["/dev/usb/lp0", "/dev/usb/lp1", "/dev/ec58_printer", "/dev/lp0"]
    printer_dev = None
    for d in devices:
        if os.path.exists(d):
            printer_dev = d
            break

    if not printer_dev:
        status_result["message"] = "Thermal printer not detected on /dev/usb/lp0 (Printer may be offline or unplugged)"
        return status_result

    try:
        from escpos.printer import File
        p = File(printer_dev)

        # 1. Header (Centered, 2x width/height bold)
        p.set(align="center", bold=True, width=2, height=2)
        p.text("I.T.S\n")
        p.set(align="center", bold=True, width=1, height=1)
        p.text("GRAIN AGE & QUALITY ANALYZER\n")
        p.text("--------------------------------\n")

        # 2. Sample Details
        p.set(align="left", bold=False, width=1, height=1)
        p.text(f"Date/Time: {now_str}\n")
        p.text(f"Batch No : {batch_no}\n")
        p.text(f"Total Grn: {total_grains}\n")
        p.text("--------------------------------\n")
        
        # Table Header
        p.set(bold=True)
        p.text("PARAM            CNT   %   LIMIT\n")
        p.set(bold=False)
        p.text("--------------------------------\n")

        # Table rows
        for s in stats:
            name = s.get("param", "")
            short_names = {
                "Good / Head Rice": "Head Rice",
                "Big Brokens (25-75%)": "Big Brokens",
                "Small Brokens (<25%)": "Small Brokens",
                "Foreign Matter": "Foreign Mat",
                "Damaged Grains": "Damaged",
                "Discoloured Grains": "Discoloured",
                "Chalky Grains": "Chalky",
                "Unresolved Clusters / Review Pending": "Clusters"
            }
            p_name = short_names.get(name, name[:14]).ljust(14)[:14]
            cnt = str(s.get("count", 0)).rjust(4)[:4]
            pct_val = s.get("mass_pct", s.get("pct", 0.0))
            pct_str = f"{pct_val:.1f}%".rjust(5)[:5]
            lim_clean = s.get("limit", "-").replace("Min ", ">=").replace("Max ", "<=").replace(".00%", "%").replace(" (Informational)", "")[:6].rjust(6)
            p.text(f"{p_name} {cnt} {pct_str} {lim_clean}\n")

        p.text("--------------------------------\n")
        p.set(bold=True)
        p.text("KERNEL GEOMETRY & AGE\n")
        p.set(bold=False)
        p.text(f"Avg Length  : {avg_l:.3f} mm\n")
        p.text(f"Avg Breadth : {avg_b:.3f} mm\n")
        p.text(f"L/B Ratio   : {lb:.2f} ({grade})\n")
        p.text(f"Chemical Age: {age_str}\n")
        p.text("================================\n")

        # Verdict
        p.set(align="center", bold=True, width=2, height=2)
        p.text(f"{disposition.upper()}\n")

        # Footer
        p.set(align="center", bold=False, width=1, height=1)
        p.text("================================\n")
        p.text("* Graded per AP Civil Supplies\n")
        p.text("  & Codex CXS 198 Standards\n\n")

        # QR Code with link to result
        qr_url = f"http://192.168.23.142:5000"
        try:
            p.qr(qr_url, size=5)
            p.text("\n")
        except Exception:
            pass

        # Feed past manual tear bar
        p.text("\n\n\n")
        p.close()

        status_result["success"] = True
        status_result["device"] = printer_dev
        status_result["message"] = f"Receipt printed successfully to {printer_dev}!"

    except Exception as e:
        status_result["success"] = False
        status_result["message"] = f"Print error: {str(e)}"

    return status_result
