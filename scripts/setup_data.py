"""
Setup script: tạo data cho Task 1 & 2.
Chạy: python scripts/setup_data.py
"""

import json
import struct
import zlib
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
NEWS_DIR = PROJECT_DIR / "data" / "landing" / "news"


def _make_pdf(text: str, title: str) -> bytes:
    """Tạo PDF đơn giản với nội dung text (không cần thư viện ngoài)."""
    lines = [title, ""] + text.split("\n")
    content_lines = ["BT", "/F1 11 Tf", "50 750 Td"]
    y = 0
    for line in lines[:80]:
        safe = (
            line.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .encode("latin-1", errors="replace")
            .decode("latin-1")
        )
        if y > 0:
            content_lines.append("0 -14 Td")
        content_lines.append(f"({safe}) Tj")
        y += 14
    content_lines.append("ET")
    stream = "\n".join(content_lines)
    stream_bytes = stream.encode("latin-1", errors="replace")

    objects = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    objects.append(
        f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj

    xref_pos = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return pdf


def _make_docx(text: str, title: str) -> bytes:
    """Tạo DOCX tối thiểu (ZIP + XML) không cần python-docx."""
    import zipfile
    import io

    paragraphs = [title] + text.split("\n")
    body = ""
    for p in paragraphs:
        safe = (
            p.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        body += f"<w:p><w:r><w:t>{safe}</w:t></w:r></w:p>"

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}<w:sectPr/></w:body>
</w:document>"""

    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


LEGAL_DOCS = {
    "luat-phong-chong-ma-tuy-2021.pdf": """LUẬT PHÒNG, CHỐNG MA TÚY 2021 (Luật số 73/2021/QH15)

Chương I: NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Luật này quy định về phòng ngừa, phát hiện, đấu tranh, xử lý các hành vi vi phạm pháp luật
về ma túy; cai nghiện ma túy; quản lý, giáo dục sau cai nghiện; trách nhiệm của cơ quan,
tổ chức, cá nhân trong phòng, chống ma túy.

Điều 2. Đối tượng áp dụng
Luật này áp dụng đối với cơ quan, tổ chức, cá nhân Việt Nam và cơ quan, tổ chức, cá nhân
nước ngoài có hoạt động liên quan đến phòng, chống ma túy trên lãnh thổ Việt Nam.

Điều 3. Giải thích từ ngữ
1. Ma túy là chất ma túy và tiền chất ma túy.
2. Chất ma túy là chất kích thích thần kinh có tác dụng gây nghiện, ảo giác.
3. Tiền chất ma túy là chất tham gia vào quá trình tổng hợp chất ma túy.

Chương II: PHÒNG NGỪA MA TÚY

Điều 15. Giáo dục phòng chống ma túy
Nhà nước tổ chức giáo dục phòng chống ma túy trong nhà trường, cộng đồng, gia đình.

Chương III: CAI NGHIỆN MA TÚY

Điều 40. Hình thức cai nghiện ma túy
1. Cai nghiện ma túy tự nguyện tại cộng đồng.
2. Cai nghiện ma túy tại gia đình.
3. Cai nghiện ma túy bắt buộc tại cơ sở cai nghiện bắt buộc.

Điều 41. Đối tượng cai nghiện bắt buộc
Người từ đủ 12 tuổi đến dưới 18 tuổi sử dụng trái phép chất ma túy.
Người từ đủ 18 tuổi sử dụng trái phép chất ma túy lần thứ hai trở lên.""",

    "nghi-dinh-105-2021.pdf": """NGHỊ ĐỊNH 105/2021/NĐ-CP
Hướng dẫn thi hành Luật Phòng, chống ma túy

Chương I: QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Nghị định này hướng dẫn chi tiết thi hành Luật Phòng, chống ma túy về phòng ngừa,
phát hiện, đấu tranh, xử lý vi phạm, cai nghiện ma túy.

Điều 2. Danh mục chất ma túy
Danh mục chất ma túy, tiền chất ma túy được ban hành kèm theo Nghị định này,
bao gồm các nhóm chất theo Công ước Liên Hợp Quốc.

Chương II: XỬ LÝ VI PHẠM HÀNH CHÍNH

Điều 10. Xử phạt vi phạm hành chính về ma túy
Phạt tiền từ 500.000 đồng đến 2.000.000 đồng đối với hành vi sử dụng trái phép
chất ma túy lần đầu.
Tịch thu tang vật, phương tiện vi phạm theo quy định của pháp luật.

Chương III: TỔ CHỨC THỰC HIỆN

Điều 20. Trách nhiệm của Bộ Công an
Bộ Công an chủ trì, phối hợp với các bộ, ngành trong công tác phòng, chống ma túy.""",

    "bo-luat-hinh-su-dieu-248.docx": """BỘ LUẬT HÌNH SỰ 2015 (sửa đổi 2017)
Chương XX: CÁC TỘI PHẠM VỀ MA TÚY

Điều 247. Tội sản xuất trái phép chất ma túy
1. Người nào sản xuất trái phép chất ma túy thuộc Danh mục I, bị phạt tù từ 15 năm
đến 20 năm, tù chung thân hoặc tử hình.
2. Trường hợp sản xuất từ 100 gram trở lên, phạt tù từ 20 năm đến tù chung thân
hoặc tử hình.

Điều 248. Tội tàng trữ trái phép chất ma túy
1. Người nào tàng trữ trái phép chất ma túy thuộc Danh mục I, bị phạt tù từ 02 năm
đến 07 năm.
2. Tàng trữ từ 5 gram đến dưới 100 gram, phạt tù từ 07 năm đến 15 năm.
3. Tàng trữ từ 100 gram trở lên, phạt tù từ 15 năm đến 20 năm hoặc tù chung thân.

Điều 249. Tội mua bán trái phép chất ma túy
1. Người nào mua bán trái phép chất ma túy thuộc Danh mục I, bị phạt tù từ 15 năm
đến 20 năm, tù chung thân hoặc tử hình.

Điều 250. Tội vận chuyển trái phép chất ma túy
Người vận chuyển trái phép chất ma túy bị phạt tù từ 07 năm đến 15 năm.

Điều 251. Tội tổ chức, cầm đầu đường dây mua bán trái phép chất ma túy
Phạt tù từ 20 năm đến tù chung thân hoặc tử hình.

Điều 252. Tội sử dụng trái phép chất ma túy
Người sử dụng trái phép chất ma túy bị xử lý hành chính hoặc đưa vào cơ sở cai nghiện.""",
}

NEWS_ARTICLES = [
    {
        "url": "https://vnexpress.net/nghe-si-bi-bat-vi-ma-tuy-2024",
        "title": "Ca sĩ nổi tiếng bị bắt vì tàng trữ ma túy tại TP.HCM",
        "content_markdown": """# Ca sĩ nổi tiếng bị bắt vì tàng trữ ma túy tại TP.HCM

Công an TP.HCM ngày 15/3/2024 đã bắt giữ ca sĩ H.T. (35 tuổi) tại căn hộ cao cấp
ở quận 2 vì hành vi tàng trữ trái phép chất ma túy.

Theo kết quả điều tra ban đầu, tang vật thu giữ gồm 2 gói tinh thể màu trắng
nghi là ma túy tổng hợp, cân nặng khoảng 5 gram. Ca sĩ thừa nhận sử dụng ma túy
để "giảm stress" trong quá trình biểu diễn.

Luật sư cho biết hành vi này có thể bị xử lý theo Điều 248 Bộ luật Hình sự với
hình phạt tù từ 2 đến 7 năm. Nghệ sĩ đang bị tạm giữ để điều tra thêm.""",
    },
    {
        "url": "https://tuoitre.vn/dien-vien-bi-khoi-to-ma-tuy",
        "title": "Diễn viên điện ảnh bị khởi tố vì sử dụng ma túy",
        "content_markdown": """# Diễn viên điện ảnh bị khởi tố vì sử dụng ma túy

Viện Kiểm sát Nhân dân TP.Hà Nội ngày 20/6/2023 đã phê chuẩn quyết định khởi tố
bị can đối với diễn viên N.V.A. (28 tuổi) về tội tàng trữ trái phép chất ma túy.

Diễn viên bị phát hiện tại một quán bar ở quận Hoàn Kiếm trong đợt kiểm tra
định kỳ của công an. Test nhanh cho kết quả dương tính với chất ma túy.

Theo Luật Phòng, chống ma túy 2021, người sử dụng trái phép chất ma túy lần đầu
có thể bị xử phạt hành chính hoặc đưa vào cơ sở cai nghiện bắt buộc.""",
    },
    {
        "url": "https://thanhnien.vn/rapper-bi-phat-tu-ma-tuy",
        "title": "Rapper trẻ lĩnh 4 năm tù vì tàng trữ ma túy",
        "content_markdown": """# Rapper trẻ lĩnh 4 năm tù vì tàng trữ ma túy

TAND quận 7, TP.HCM ngày 10/1/2024 tuyên phạt rapper L.K. (24 tuổi) 4 năm tù
về tội tàng trữ trái phép chất ma túy theo Điều 248 Bộ luật Hình sự.

Bị cáo bị bắt quả tang tại studio thu âm với 3,2 gram ma túy tổng hợp loại ketamine.
Trong phiên tòa, rapper thừa nhận mua ma túy qua mạng xã hội để sử dụng cá nhân.

Hội đồng xét xử nhận định đây là lần đầu phạm tội nhưng cần có hình phạt nghiêm
để răn đe, tuy nhiên giảm nhẹ một phần vì bị cáo ăn năn.""",
    },
    {
        "url": "https://vnexpress.net/dj-bi-bat-tai-le-hoi-am-nhac",
        "title": "DJ nổi tiếng bị bắt tại lễ hội âm nhạc vì ma túy",
        "content_markdown": """# DJ nổi tiếng bị bắt tại lễ hội âm nhạc vì ma túy

Công an tỉnh Bình Dương ngày 5/12/2023 đã bắt giữ DJ T.M. (30 tuổi) ngay sau
buổi biểu diễn tại một lễ hội âm nhạc lớn.

Tang vật thu giữ gồm 10 viên thuốc lắc và một túi bột trắng nghi là ma túy.
DJ khai nhận mua ma túy từ một người bạn trong giới giải trí.

Sự việc gây chấn động dư luận khi DJ này từng là gương mặt đại diện cho nhiều
thương hiệu lớn. Các nhãn hàng đã lập tức chấm dứt hợp đồng quảng cáo.""",
    },
    {
        "url": "https://tuoitre.vn/model-bi-dua-vao-cai-nghien",
        "title": "Người mẫu được đưa vào cơ sở cai nghiện bắt buộc",
        "content_markdown": """# Người mẫu được đưa vào cơ sở cai nghiện bắt buộc

UBND quận 1, TP.HCM ngày 8/9/2023 ra quyết định đưa người mẫu P.T.H. (26 tuổi)
vào cơ sở cai nghiện bắt buộc sau khi bị phát hiện sử dụng ma túy lần thứ ba.

Theo Luật Phòng, chống ma túy 2021, người từ đủ 18 tuổi sử dụng trái phép chất
ma túy lần thứ hai trở lên sẽ bị đưa vào cơ sở cai nghiện bắt buộc.

Người mẫu sẽ phải cai nghiện trong thời gian từ 12 đến 24 tháng tại Trung tâm
Giáo dục lao động - xã hội số 2 TP.HCM.""",
    },
]


def setup_legal_docs():
    LEGAL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in LEGAL_DOCS.items():
        path = LEGAL_DIR / filename
        if path.exists() and path.stat().st_size > 1024:
            print(f"  skip (exists): {filename}")
            continue
        title = content.split("\n")[0]
        if filename.endswith(".pdf"):
            data = _make_pdf(content, title)
        else:
            data = _make_docx(content, title)
        path.write_bytes(data)
        print(f"  created: {filename} ({len(data)} bytes)")


def setup_news_articles():
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    for i, article in enumerate(NEWS_ARTICLES, 1):
        filename = f"article_{i:02d}.json"
        path = NEWS_DIR / filename
        data = {
            "url": article["url"],
            "title": article["title"],
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": article["content_markdown"],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  created: {filename}")


if __name__ == "__main__":
    print("Task 1: Legal documents")
    setup_legal_docs()
    print("\nTask 2: News articles")
    setup_news_articles()
    print("\nDone!")
