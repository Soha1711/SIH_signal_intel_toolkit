"""
audit_modules.py  --  STEP 1 helper
Lists every public function/class (with arguments and return keys it appears to build)
in the module folders WITHOUT importing them, so it works even if a module is broken.

Run from the repo root:   python audit_modules.py
Output: prints a report and writes docs/module_audit.md
"""
import ast
import os

FOLDERS = ["ingestion", "param_estimation", "demodulation", "deinterleave_fec", "gui"]
CONTRACT_KEYS = {
    "samples", "sample_rate", "source_format", "duration_sec", "modulation_type",
    "symbol_rate", "occupied_bandwidth", "confidence", "bitstream", "symbol_stream",
    "deinterleaved_bits", "interleaver_type", "decoded_bits", "fec_scheme", "error_count",
    "header_start", "header_end", "payload_start", "payload_end", "sync_word_matched",
}


def string_keys(node):
    """All string literals that look like dict keys inside a function body."""
    found = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    found.add(k.value)
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
            found.add(n.slice.value)
    return found


def describe(path):
    rows = []
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except SyntaxError as exc:
        return [("!! SYNTAX ERROR", str(exc), "")]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            args = ", ".join(a.arg for a in node.args.args)
            keys = sorted(string_keys(node) & CONTRACT_KEYS)
            rows.append((f"def {node.name}({args})", ", ".join(keys), ast.get_docstring(node) or ""))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
            keys = sorted(string_keys(node) & CONTRACT_KEYS)
            rows.append((f"class {node.name}: {', '.join(methods)}", ", ".join(keys), ast.get_docstring(node) or ""))
    return rows


def main():
    lines = ["# Module audit (auto-generated)\n",
             "Fill the last column by hand: does it follow docs/interface_contract.md? What is missing?\n"]
    for folder in FOLDERS:
        lines.append(f"\n## {folder}/\n")
        if not os.path.isdir(folder):
            lines.append("_folder not found_\n")
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".py") or fname == "__init__.py":
                continue
            rows = describe(os.path.join(folder, fname))
            lines.append(f"\n### {folder}/{fname}\n")
            if not rows:
                lines.append("_no public functions/classes_\n")
                continue
            lines.append("| Signature | Contract keys mentioned | Follows contract? / notes |\n|---|---|---|\n")
            for sig, keys, _doc in rows:
                lines.append(f"| `{sig}` | {keys or '-'} |  |\n")
    report = "".join(lines)
    os.makedirs("docs", exist_ok=True)
    with open("docs/module_audit.md", "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print("\nSaved to docs/module_audit.md")


if __name__ == "__main__":
    main()