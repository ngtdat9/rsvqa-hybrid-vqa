#!/usr/bin/env bash
# Rebuild every README figure and the progress report PDF (macOS, needs Google Chrome).
#
#     bash docs/report/build.sh
#
# 1. charts: scripts/make_figures.py -> reports/figures/readme/*.png and docs/report/charts/*.svg
# 2. diagrams and the key-results card -> reports/figures/readme/*.png
# 3. docs/report/report.html -> reports/RSVQA_progress_report_weeks1-6.pdf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPORT="$ROOT/docs/report"
README_FIGS="$ROOT/reports/figures/readme"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
chrome() { "$CHROME" --headless=new --disable-gpu --hide-scrollbars --allow-file-access-from-files "$@" >/dev/null 2>&1; }

python "$ROOT/scripts/make_figures.py"

# diagrams (700 wide; heights from their viewBox) and the results card, at 2x
chrome --force-device-scale-factor=2 --window-size=700,300 --screenshot="$README_FIGS/architecture.png" "file://$REPORT/diagrams/architecture.svg"
chrome --force-device-scale-factor=2 --window-size=700,250 --screenshot="$README_FIGS/rag_pipeline.png" "file://$REPORT/diagrams/rag_pipeline.svg"
chrome --force-device-scale-factor=2 --window-size=900,128 --screenshot="$README_FIGS/key_results.png" "file://$REPORT/kpis.html"

# the PDF: placeholders in report.html become absolute file paths in a temporary copy
BUILT="$(mktemp -t report).html"
sed -e "s#src=\"FIG/#src=\"file://$ROOT/reports/figures/#g" \
    -e "s#src=\"CH/#src=\"file://$REPORT/charts/#g" \
    -e "s#src=\"DIAG/#src=\"file://$REPORT/diagrams/#g" "$REPORT/report.html" > "$BUILT"
chrome --no-pdf-header-footer --print-to-pdf="$ROOT/reports/RSVQA_progress_report_weeks1-6.pdf" "file://$BUILT"
rm -f "$BUILT"
echo "built: reports/RSVQA_progress_report_weeks1-6.pdf and reports/figures/readme/"
