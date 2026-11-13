from pdf2image import convert_from_path
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert PDF to PNG images')
parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the PNG images')
args = parser.parse_args()

pages = convert_from_path(args.pdf_path, dpi=600)

output_dir = Path(args.output_dir)
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

for i, page in enumerate(pages):
    page.save(output_dir / f"original_{i+1}.png")
    print(f"page_{i+1}.png を保存しました")
