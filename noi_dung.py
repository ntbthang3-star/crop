"""Nội dung cây trồng và lời khuyên (dùng chung cho app.py)."""

LBL = {"N": "N - Đạm (Nitrogen)", "P": "P - Lân (Phosphorus)", "K": "K - Kali (Potassium)",
       "temperature": "Nhiệt độ (°C)", "humidity": "Độ ẩm không khí (%)", "pH": "pH - Độ chua của đất",
       "rainfall": "Lượng mưa (mm)"}
SHORT = {"N": "N", "P": "P", "K": "K", "temperature": "Nhiệt độ", "humidity": "Độ ẩm", "pH": "pH", "rainfall": "Lượng mưa"}
LIMIT = {"N": (0.0, 300.0, 1.0), "P": (0.0, 300.0, 1.0), "K": (0.0, 400.0, 1.0), "temperature": (0.0, 50.0, 0.5),
         "humidity": (0.0, 100.0, 1.0), "pH": (0.0, 14.0, 0.1), "rainfall": (0.0, 600.0, 1.0)}

# key trong dataset: (tên Việt, tên khoa học, nhóm, emoji, thời gian, ghi chú)
CROPS = {
    "rice": ("Lúa", "Oryza sativa", "Ngũ cốc", "90–120 ngày", "Cây lương thực chính, ưa nóng ẩm và nhiều nước."),
    "maize": ("Ngô", "Zea mays", "Ngũ cốc", "80–110 ngày", "Cây ngắn ngày, nhạy cảm với úng và hạn lúc trổ cờ."),
    "chickpea": ("Đậu gà", "Cicer arietinum", "Đậu đỗ", "90–120 ngày", "Chịu hạn tốt, hợp mùa khô mát."),
    "kidneybeans": ("Đậu đỏ", "Phaseolus vulgaris", "Đậu đỗ", "60–120 ngày", "Cần đất tơi xốp, không chịu được úng."),
    "pigeonpeas": ("Đậu triều", "Cajanus cajan", "Đậu đỗ", "120–180 ngày", "Cây chịu hạn, rễ sâu, cải tạo đất tốt."),
    "mothbeans": ("Đậu mỡ", "Vigna aconitifolia", "Đậu đỗ", "60–90 ngày", "Rất chịu nóng và khô, hợp vùng ít mưa."),
    "mungbean": ("Đậu xanh", "Vigna radiata", "Đậu đỗ", "60–75 ngày", "Cây ngắn ngày, hợp xen canh sau lúa."),
    "blackgram": ("Đậu đen", "Vigna mungo", "Đậu đỗ", "70–90 ngày", "Ưa ấm, ẩm vừa, cố định đạm cho đất."),
    "lentil": ("Đậu lăng", "Lens culinaris", "Đậu đỗ", "80–120 ngày", "Ưa khí hậu mát, khô ráo."),
    "pomegranate": ("Lựu", "Punica granatum", "Cây ăn quả", "2–3 năm cho quả đầu", "Chịu hạn khá, cần nắng nhiều để quả chín đẹp."),
    "banana": ("Chuối", "Musa spp.", "Cây ăn quả", "9–12 tháng thu hoạch", "Cần ẩm cao, nhiều nước và nhiều kali."),
    "mango": ("Xoài", "Mangifera indica", "Cây ăn quả", "3–5 năm cho quả đầu", "Ưa nóng, cần mùa khô ngắn để ra hoa tập trung."),
    "grapes": ("Nho", "Vitis vinifera", "Cây ăn quả", "2–3 năm cho quả", "Cần nắng, thoát nước tốt và giàn leo."),
    "apple": ("Táo", "Malus domestica", "Cây ăn quả", "2–5 năm cho quả", "Cần khí hậu mát, có giai đoạn lạnh để ra hoa."),
    "orange": ("Cam", "Citrus sinensis", "Cây ăn quả", "2–3 năm cho quả đầu", "Ưa ấm, đất thoát nước, tưới đều."),
    "papaya": ("Đu đủ", "Carica papaya", "Cây ăn quả", "8–10 tháng", "Lớn nhanh, sợ úng, cần ấm quanh năm."),
    "coconut": ("Dừa", "Cocos nucifera", "Cây ăn quả", "5–7 năm cho quả đầu", "Hợp vùng ven biển, ẩm cao, mưa nhiều."),
    "watermelon": ("Dưa hấu", "Citrullus lanatus", "Dưa", "60–90 ngày", "Cần nắng, đất cát pha thoát nước, ít mưa lúc chín."),
    "muskmelon": ("Dưa lê", "Cucumis melo", "Dưa", "70–90 ngày", "Ưa nắng ấm, tránh ẩm thấp gây bệnh lá."),
    "cotton": ("Bông", "Gossypium spp.", "Cây công nghiệp", "150–180 ngày", "Cần nắng nhiều, mùa thu hoạch cần khô ráo."),
    "jute": ("Đay", "Corchorus spp.", "Cây công nghiệp", "100–120 ngày", "Ưa nóng ẩm, mưa nhiều, đất phù sa."),
    "coffee": ("Cà phê", "Coffea spp.", "Cây công nghiệp", "3–4 năm cho quả đầu", "Cần mưa đủ, mùa khô ngắn để phân hóa mầm hoa."),
}
GROUPS = ["Ngũ cốc", "Đậu đỗ", "Cây ăn quả", "Dưa", "Cây công nghiệp"]
PERENNIAL = {k for k, v in CROPS.items() if v[2] == "Cây ăn quả"} | {"coffee"}
GROUP_SEASON = {"Ngũ cốc": "Trồng theo vụ; chọn vụ có đủ nắng ấm và nguồn nước.",
                "Đậu đỗ": "Thường trồng xen hoặc nối vụ sau lúa; tránh mùa mưa lớn lúc thu hoạch.",
                "Cây ăn quả": "Cây lâu năm; trồng đầu mùa mưa để cây ít phải tưới, thu hoạch theo mùa quả.",
                "Dưa": "Trồng vụ khô nắng, đủ ánh sáng để quả ngọt; tránh mưa nhiều lúc quả chín.",
                "Cây công nghiệp": "Chọn thời điểm đầu mùa mưa (cây lâu năm) hoặc đầu vụ nắng ấm (cây hằng năm)."}

STEPS_ANNUAL = [("Làm đất & bón lót", "Cày bừa tơi xốp, làm sạch cỏ, bón lót phân hữu cơ hoai mục và lân."),
                ("Gieo trồng", "Chọn giống tốt, xử lý hạt, gieo đúng mật độ và độ sâu theo giống."),
                ("Bón thúc & tưới", "Chia phân đạm, kali làm 2–3 lần; giữ ẩm đều, nhất là lúc ra hoa."),
                ("Chăm sóc & phòng bệnh", "Làm cỏ, vun gốc, theo dõi sâu bệnh định kỳ, ưu tiên biện pháp sinh học."),
                ("Thu hoạch & bảo quản", "Thu khi đạt độ chín, phơi sấy hoặc bảo quản nơi khô thoáng.")]
STEPS_PERENNIAL = [("Làm đất & đào hố", "Đào hố rộng, bón lót phân hữu cơ và lân, để đất nghỉ ít nhất 2–4 tuần."),
                   ("Trồng cây giống", "Chọn cây giống sạch bệnh, trồng đầu mùa mưa, che bóng và tưới giữ ẩm giai đoạn đầu."),
                   ("Bón phân & tưới", "Bón theo giai đoạn: kiến thiết cơ bản, trước ra hoa, nuôi quả; tưới giữ ẩm đều."),
                   ("Tỉa cành & phòng bệnh", "Tạo tán, tỉa cành thông thoáng; theo dõi sâu bệnh, vệ sinh vườn sau thu hoạch."),
                   ("Thu hoạch & chăm sóc sau thu", "Thu đúng độ chín, cắt tỉa, bón phục hồi để chuẩn bị mùa sau.")]

ADV = {"N": ("Đạm thấp hơn mức cây thường ưa. Bón thêm phân đạm hoặc phân hữu cơ hoai mục, chia làm nhiều lần.",
             "Đạm cao hơn mức điển hình. Giảm hoặc bỏ phân đạm, tránh bón thúc nhiều khiến cây vươn lá, đổ ngã và nhiều sâu bệnh."),
       "P": ("Lân thấp. Bón lót phân lân (super lân, lân nung chảy) kèm phân chuồng hoai.",
             "Lân cao hơn mức điển hình. Có thể gây thiếu kẽm, sắt; tạm ngừng bón lân."),
       "K": ("Kali thấp. Bổ sung phân kali hoặc tro bếp, nhất là giai đoạn nuôi quả, hạt.",
             "Kali cao. Giảm phân kali, theo dõi cân đối canxi và magiê để cây không thiếu chất khác."),
       "temperature": ("Nhiệt độ thấp hơn cây ưa. Chọn mùa vụ ấm hơn, dùng màng phủ hoặc che chắn gió lạnh.",
                       "Nhiệt độ cao hơn cây ưa. Tránh mùa nóng nhất, che lưới giảm nắng, tưới vào sáng sớm hoặc chiều mát."),
       "humidity": ("Không khí khô hơn cây ưa. Tưới phun sương, phủ gốc giữ ẩm, trồng cây chắn gió.",
                    "Độ ẩm cao dễ phát sinh nấm bệnh. Trồng thưa, tỉa cành thông thoáng, vệ sinh vườn."),
       "pH": ("Đất chua hơn cây ưa. Bón vôi hoặc dolomite, kết hợp phân hữu cơ để nâng pH.",
              "Đất kiềm hơn cây ưa. Dùng phân hữu cơ, phân chua sinh lý hoặc lưu huỳnh, tưới đủ nước để rửa bớt muối."),
       "rainfall": ("Lượng mưa thấp hơn nhu cầu. Cần tưới bổ sung (nhỏ giọt, phun), phủ gốc giữ ẩm.",
                    "Lượng mưa cao hơn nhu cầu. Làm luống cao, đào rãnh thoát nước, tránh úng gốc.")}
