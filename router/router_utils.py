PX_TO_M = 0.15        # 1 px ≈ 0.15 m (kendi haritana göre kalibre et)
WALK_M_PER_MIN = 84   # ~1.4 m/s

PATH_WEIGHTS = {
    "indoor":   0.8,   # iç yollar (koridorlar) tercih edilir
    "footpath": 1.0,   # normal yürüyüş yolları
    "sidewalk": 1.05,  # kaldırım
    "crosswalk":1.1,   # yol geçişi
    "stairs":   3.0    # merdiven (erişilebilir değil)
}
def px_to_m(px: float) -> float:
    return px * PX_TO_M

def px_to_min(px: float) -> float:
    return px_to_m(px) / WALK_M_PER_MIN

