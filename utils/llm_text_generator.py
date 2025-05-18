"""
summarise_folders.py  –  recursive edition
──────────────────────────────────────────
Generates LLM‑ready Markdown summaries for **every nested sub‑folder**
inside one or more root directories.
(See header in the previous version for full description.)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Iterable, List
import yaml # Add yaml import

sys.path.append("./")
sys.path.append("../")

from rich.console import Console

console = Console()

# -- Your OpenAI utilities ----------------------------------------------------
from utils.openai_utils import *
from utils.openai_data_models import *
from utils.file_utils import *

# ------------------------------------------------------------------#
#  Build *consolidated import statements* for source code   #
#  IMPORT DISCOVERY – with de‑dup & whitelist                       #
# ------------------------------------------------------------------#

import ast
from collections import defaultdict
from typing import Dict, List, Set


def _module_from_line(line: str) -> str:
    """
    Extract the *module path* token used in an import line.
        • 'from a.b.c import X'  -> 'a.b.c'
        • 'import a.b.c as foo'  -> 'a.b.c'
        • 'import a, b'          -> 'a'   (first token)
    """
    line = line.strip()
    if line.startswith("from "):
        return line.split()[1]
    if line.startswith("import "):
        # take first token after 'import', strip comma / as …
        rest = line.split(maxsplit=1)[1]
        token = re.split(r"[,\s]", rest, maxsplit=1)[0]
        return token
    return ""


def collect_import_statements(
    source_root: pathlib.Path,
    allowed_prefixes: Set[str] | None = None,
) -> str:
    """
    Return a *multiline string* containing every unique import line that
    (a) starts with 'from ' or 'import ', (b) optionally matches the
    given `allowed_prefixes`.

    Example:
        ALLOWED = {"semantic_kernel"}
        IMPORTS_FOR_PROMPT = collect_import_lines(
            pathlib.Path("semantic_kernel"),
            allowed_prefixes=ALLOWED,
        )
    """
    console.rule(f"[bold blue]Collecting raw import lines under: {source_root}")

    raw_lines: Set[str] = set()

    for py_file in source_root.rglob("*.py"):
        try:
            for line in py_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.lstrip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith(("from ", "import ")):
                    mod = _module_from_line(stripped)
                    if allowed_prefixes and not any(mod.startswith(p) for p in allowed_prefixes):
                        continue
                    raw_lines.add(stripped.rstrip())
        except Exception as exc:                     # noqa: BLE001
            console.print(f"[dim]⚠️  Skip {py_file} ({exc})")

    sorted_lines: List[str] = sorted(raw_lines, key=str.lower)
    console.print(f"[green]✓ Kept {len(sorted_lines)} unique import line(s).")
    return "\n".join(sorted_lines)


# -- 1. Prompt templates ------------------------------------------------------

PROMPT_TEMPLATE = """
You are a senior software‑development educator and technical writer.

## Context
The following code comes from one cohesive sub‑folder inside a large
GitHub repository.  Treat it as a *single knowledge capsule*:

{context}


## Imports 
It is noticeable that the generated output uses the wrong imports, or some made-up imports. The following import statements are present in the code capsule. Please **MAKE SURE** to use the correct imports ONLY.
{import_statements}

## Task
Produce a Markdown document that can be ingested by other LLMs.
Follow **ALL** requirements exactly:

1. **Extract and explain** every development principle, pattern, architectural style,
   library, algorithm, or domain‑specific technique present in the context.  
   • *Be verbose*: give enough depth that a mid‑level engineer can learn the why & how,  
   • *De‑duplicate*: strip boiler‑plate or repeated snippets.
2. Organise the output in **this order** (use H2 headings `##`):
   - **Overview of Concepts**
   - **Key Principles & Patterns**
   - **Illustrative Code Snippets** – concise, self‑contained; you may re‑create code.
   - **Insights**
   - **Pitfalls**
   - **Further Reading / Links** (optional)
3. Generate ONE full end‑to‑end code example:
    - **Full Code Example** – concise, self‑contained; you may re‑create code.
    - **Captures ALL Concepts** from the context.
    - **Captures ALL Principles & Patterns** from the context.
    - **Azure Models and Technologies**: give preference to the Azure AI models, SDKs and APIs.
4. Pay **special attention" to your Import statements and Function Signatures**:
    - **Imaginary Import**: do NOT make up imports
    - **Confirm the Imports**: do a double-check using the context code to make sure the Imports are correct.
    - **Include ALL Imports**: include all the imports that are used in the context code.
    - **Confirm the Function Signatures**: do a double-check using the context code to make sure the Function Signatures are correct.
5. *Formatting*:
   • No file paths, no personal commentary, no internal notes.  
   • Fence every code block with triple back‑ticks and an explicit language tag.  
   • Output **plain Markdown only** – no YAML front‑matter, no HTML.

   
Here are also additional instructions to follow that are specific to this repo:
{specific_instructions}


Begin now.
"""

SLUG_PROMPT = """
Generate a concise, filesystem‑safe slug (lowercase, kebab‑case, 3‑8 words, no spaces)
that summarises the main theme of the following Markdown document.  
Return **only** the slug, nothing else.

###
{summary}
###
"""

# -- 2. Helpers for reading code ---------------------------------------------

def read_python_file(path: pathlib.Path) -> str:
    console.print(f"[cyan]Reading {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")

def read_notebook(path: pathlib.Path) -> str:
    console.print(f"[cyan]Reading {path}")
    try:
        import nbformat
        nb = nbformat.read(path, as_version=4)
        code_cells = [
            "".join(cell.source)
            for cell in nb.cells
            if cell.get("cell_type") == "code"
        ]
        return "\n\n".join(code_cells)
    except Exception:                        # noqa: BLE001
        return path.read_text(encoding="utf-8", errors="ignore")

def gather_code(folder: pathlib.Path) -> str:
    parts: List[str] = []
    for file in folder.rglob("*"):
        if file.suffix == ".py":
            parts.append(read_python_file(file))
        elif file.suffix == ".ipynb":
            parts.append(read_notebook(file))
    return "\n\n# ─────────────────────────────────────────────\n\n".join(parts)

def folder_has_code(folder: pathlib.Path) -> bool:
    return any(
        p.suffix in {".py", ".ipynb"} for p in folder.rglob("*")
    )

# -- 3. LLM wrappers ----------------------------------------------------------

def call_text_llm(prompt: str, model_info: str, reasoning_efforts: str) -> str:
    cfg = TextProcessingModelnfo(model_name=model_info, reasoning_efforts=reasoning_efforts)
    return call_llm(prompt, model_info=cfg)          # type: ignore  # noqa: F821

def generate_slug(summary_md: str, model_info: str, reasoning_efforts: str) -> str:
    raw = call_text_llm(
        SLUG_PROMPT.format(summary=summary_md),
        model_info=model_info,
        reasoning_efforts=reasoning_efforts,
    ).strip()
    slug = re.sub(r"[^a-z0-9\-]+", "", raw.lower())
    return slug or "untitled-summary"

# -- 4. Walk every nested sub‑folder -----------------------------------------

def walk_subfolders(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield *all* sub‑directories at any depth, excluding root itself."""
    for p in root.rglob("*"):
        if p.is_dir() and p != root and not p.name.startswith("."):
            yield p

# -- 5. Main summarisation routine ------------------------------------------

def summarise_folder(
    folder: pathlib.Path,
    model_info: str,
    reasoning_efforts: str,
    import_statements: str,
    out_root: pathlib.Path = pathlib.Path("summaries"),
    out_sub_root: pathlib.Path = pathlib.Path("repo"),
    specific_instruction_file: str = "",
) -> None:
    console.rule(f"[bold yellow]{folder.relative_to(folder.anchor)}")
    if not folder_has_code(folder):
        console.print("[dim]No Python / notebook content – skipped.")
        return

    context = gather_code(folder)
    specific_instructions = read_file(specific_instruction_file) if specific_instruction_file else "No additional instructions provided."
    prompt = PROMPT_TEMPLATE.format(context=context, 
                                    import_statements=import_statements, 
                                    specific_instructions=specific_instructions
                                    )
    summary_md = call_text_llm(prompt, model_info, reasoning_efforts)

    # slug = generate_slug(summary_md, model_info, reasoning_efforts)
    slug = str(folder).replace("\\", "/").replace("//", "-").replace("./", "").replace("/", "-").replace(" ", "-").lower() 
    out_dir = out_root / out_sub_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{slug}.txt"
    out_file.write_text(summary_md, encoding="utf-8")

    full_text = ""
    for text_file in out_dir.rglob("*.txt"):
        full_text += f"\n---\nFile: {text_file}\n---\n" + read_file(text_file) + "\n\n"

    out_full_file = out_root / f"{str(out_sub_root)}_full.txt"
    out_full_file.write_text(full_text, encoding="utf-8")

    console.print(f"[green]✅  Saved {out_file}")
    console.print(f"[green]✅  Saved {out_full_file}")


# -- 6. CLI ------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM‑ready summaries for every folder that contains code — "
                    "the root path *and* all nested sub‑folders."
    )
    # Add argument for config file
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("llm_config.yaml"), help="Path to the YAML configuration file.")
    
    # Keep other arguments to allow overrides, but make them not required
    parser.add_argument("roots", nargs="*", type=pathlib.Path, help="Root directories to scan (overrides config).")
    parser.add_argument("--model", help="LLM model to use (overrides config).")
    parser.add_argument("--reasoning_efforts", help="Reasoning efforts for the LLM (overrides config).")
    parser.add_argument("--source_code", help="Path to the source code directory (overrides config).")
    parser.add_argument("--whitelist", help="List of comma-separated allowed prefixes for import statements (overrides config).")
    parser.add_argument("--out_root", help="Output root directory for summaries (overrides config).")
    parser.add_argument("--out_sub_root", help="Output sub-root directory for summaries (overrides config).")
    parser.add_argument("--specific_instruction_file", help="Path to the specific instruction file (overrides config).")
    args = parser.parse_args(argv)

    # Load config from YAML file
    config = {}
    if args.config.exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        console.print(f"[yellow]⚠️ Config file not found at {args.config}, using defaults and command-line arguments.")

    # Determine values, giving precedence to command-line args, then YAML, then defaults
    roots = args.roots if args.roots else config.get("roots", [])
    model = args.model if args.model else config.get("model", "o4-mini")
    reasoning_efforts = args.reasoning_efforts if args.reasoning_efforts else config.get("reasoning_efforts", "high")
    out_root = args.out_root if args.out_root else config.get("out_root", "summaries")
    out_sub_root = args.out_sub_root if args.out_sub_root else config.get("out_sub_root", "repo")
    source_code_path_str = args.source_code if args.source_code else config.get("source_code", "")
    whitelist_str = args.whitelist if args.whitelist else ",".join(config.get("whitelist", []))
    specific_instruction_file = args.specific_instruction_file if args.specific_instruction_file else config.get("specific_instruction_file", "")

    if not roots:
        console.print("[red]❌ No root directories specified either in config file or as command-line arguments.")
        sys.exit(1)

    # Convert root paths from string to Path objects if they came from YAML
    roots = [pathlib.Path(r) for r in roots]

    allowed_prefixes = set([x.strip() for x in whitelist_str.split(",") if x.strip()])

    src_root = pathlib.Path(source_code_path_str) if source_code_path_str else None # adjust as needed
    imports = ""
    if src_root and src_root.exists():
        imports = collect_import_statements(src_root, allowed_prefixes=allowed_prefixes)
        print(imports)
    elif source_code_path_str:
        console.print(f"[yellow]⚠️ Source code path '{source_code_path_str}' not found. Imports will not be collected.")


    # sanity check
    console.print("Paths to scan:")
    for p in roots:
        console.print(f" • {p}")

    missing = [p for p in roots if not p.exists()]
    if missing:
        console.print("[red]❌  These paths do not exist:")
        for p in missing:
            console.print(f" • {p}")
        sys.exit(1)

    for root in roots:
        console.rule(f"[bold magenta]ROOT: {root}")

        # ➊ summarise the root folder itself (if it has .py / .ipynb)
        summarise_folder(
            root,
            model_info=model,
            reasoning_efforts=reasoning_efforts,
            import_statements=imports,
            out_root=pathlib.Path(out_root) if out_root else pathlib.Path("summaries"),
            out_sub_root=pathlib.Path(out_sub_root) if out_sub_root else pathlib.Path("repo"),
            specific_instruction_file=specific_instruction_file
        )

        # ➋ then walk and summarise every descendant directory
        for sub in walk_subfolders(root):
            summarise_folder(
                sub,
                model_info=model,
                reasoning_efforts=reasoning_efforts,
                import_statements=imports,
                out_root=pathlib.Path(args.out_root) if args.out_root else pathlib.Path("summaries"),
                out_sub_root=pathlib.Path(args.out_sub_root) if args.out_sub_root else pathlib.Path("repo"),
                specific_instruction_file=specific_instruction_file
            )

if __name__ == "__main__":
    main()
