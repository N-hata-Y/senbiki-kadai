from pdf2image import convert_from_path

pages = convert_from_path('JPAN_original_kmu.pdf', dpi=600)

for i, page in enumerate(pages):
    page.save(f"original_{i+1}.png")
    print(f"page_{i+1}.png を保存しました")
