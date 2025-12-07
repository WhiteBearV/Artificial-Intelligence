# probability_fruit_box.py
# Pattern Recognition - Probability Theory (Fruit + Box)

def main():
    print("=== Pattern Recognition - Probability Theory ===")
    print("Model: กล่องแดง / กล่องน้ำเงิน + แอปเปิล / ส้ม\n")

    # ---------- [1] PRIOR: ความน่าจะเป็นของการหยิบกล่อง ----------
    # ให้ผู้ใช้กำหนดเอง (ใส่เป็นเปอร์เซ็นต์)
    p_red_percent = float(input("ใส่โอกาสหยิบ 'กล่องแดง' (%), เช่น 40 : "))
    p_blue_percent = float(input("ใส่โอกาสหยิบ 'กล่องน้ำเงิน' (%), เช่น 60 : "))

    p_red = p_red_percent / 100.0
    p_blue = p_blue_percent / 100.0

    if abs(p_red + p_blue - 1.0) > 1e-6:
        print("\n[WARN] p(red) + p(blue) != 1. จะทำการ normalize ให้อัตโนมัติ\n")
        s = p_red + p_blue
        p_red /= s
        p_blue /= s

    # ---------- [2] จำนวนผลไม้ในแต่ละกล่อง ----------
    print("\nกรอกจำนวนผลไม้ใน 'กล่องแดง'")
    red_apple = int(input("  จำนวน Apple ในกล่องแดง : "))
    red_orange = int(input("  จำนวน Orange ในกล่องแดง : "))

    print("\nกรอกจำนวนผลไม้ใน 'กล่องน้ำเงิน'")
    blue_apple = int(input("  จำนวน Apple ในกล่องน้ำเงิน : "))
    blue_orange = int(input("  จำนวน Orange ในกล่องน้ำเงิน : "))

    # ตรวจ total ของแต่ละกล่อง
    total_red = red_apple + red_orange
    total_blue = blue_apple + blue_orange

    if total_red == 0 or total_blue == 0:
        print("\n[ERROR] กล่องใดกล่องหนึ่งไม่มีผลไม้เลย (total = 0)")
        return

    # ---------- Likelihood: p(F = a | box), p(F = o | box) ----------
    p_a_given_red = red_apple / total_red
    p_o_given_red = red_orange / total_red

    p_a_given_blue = blue_apple / total_blue
    p_o_given_blue = blue_orange / total_blue

    # ---------- Marginal: p(F = a), p(F = o) ----------
    # p(F = a) = p(a|red)p(red) + p(a|blue)p(blue)
    p_F_a = p_a_given_red * p_red + p_a_given_blue * p_blue
    # p(F = o) = p(o|red)p(red) + p(o|blue)p(blue)
    p_F_o = p_o_given_red * p_red + p_o_given_blue * p_blue

    # ---------- Posterior: p(red | F = o) ด้วย Bayes ----------
    # p(red|o) = p(o|red)p(red) / p(o)
    if p_F_o == 0:
        p_red_given_o = 0.0
    else:
        p_red_given_o = (p_o_given_red * p_red) / p_F_o

    # ---------- OUTPUT ----------
    print("\n========== RESULT ==========")
    print(f"p(Box = red)   = {p_red:.4f}")
    print(f"p(Box = blue)  = {p_blue:.4f}\n")

    print(f"p(F = a | red)   = {p_a_given_red:.4f}")
    print(f"p(F = o | red)   = {p_o_given_red:.4f}")
    print(f"p(F = a | blue)  = {p_a_given_blue:.4f}")
    print(f"p(F = o | blue)  = {p_o_given_blue:.4f}\n")

    print(f"p(F = a) (Apple)  = {p_F_a:.4f}")
    print(f"p(F = o) (Orange) = {p_F_o:.4f}\n")

    print(f"p(Box = red | F = o) = {p_red_given_o:.4f}")
    print(" (ความน่าจะเป็นที่ลูกส้มที่หยิบมา มาจากกล่องสีแดง)\n")

if __name__ == "__main__":
    main()
