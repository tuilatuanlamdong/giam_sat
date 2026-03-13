# csv_db.py
import os
import csv
import time
import numpy as np
import config

CSV_PATH = config.CSV_PATH
EMB_DIR = config.EMB_DIR
SNAP_DIR = config.SNAP_DIR
FIELDNAMES = config.FIELDNAMES


def tao_db_csv():
    os.makedirs(EMB_DIR, exist_ok=True)
    os.makedirs(SNAP_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()


def tai_tat_ca_csv():
    tao_db_csv()
    ds = []

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            emb_path = (row.get("emb_file") or "").strip()
            emb = None

            if emb_path and os.path.exists(emb_path):
                try:
                    emb = np.load(emb_path).astype(np.float32)
                except Exception:
                    emb = None

            ds.append({
                "person_id": int(row["person_id"]),
                "ho_ten": row.get("ho_ten", ""),
                "ma_nv": row.get("ma_nv", ""),
                "bo_phan": row.get("bo_phan", ""),
                "ngay_sinh": row.get("ngay_sinh", ""),
                "emb_file": emb_path,
                "embed": emb
            })

    return ds


def ghi_lai_csv(ds):
    tao_db_csv()

    ds = sorted(ds, key=lambda p: int(p["person_id"]))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()

        for p in ds:
            w.writerow({
                "person_id": int(p["person_id"]),
                "ho_ten": p.get("ho_ten", ""),
                "ma_nv": p.get("ma_nv", ""),
                "bo_phan": p.get("bo_phan", ""),
                "ngay_sinh": p.get("ngay_sinh", ""),
                "emb_file": p.get("emb_file", ""),
            })


def next_person_id(ds):
    if not ds:
        return 1
    return max(int(p["person_id"]) for p in ds) + 1


def person_id_exists(ds, person_id):
    for p in ds:
        if int(p["person_id"]) == int(person_id):
            return True
    return False


def them_nhan_su_csv(person_id, ho_ten, ma_nv, bo_phan, ngay_sinh, embed):
    """
    Nếu person_id để trống -> tự tăng.
    Nếu nhập person_id -> kiểm tra trùng.
    """
    ds = tai_tat_ca_csv()

    if person_id is None or str(person_id).strip() == "":
        new_id = next_person_id(ds)
    else:
        new_id = int(person_id)
        if person_id_exists(ds, new_id):
            print(f"[DK] person_id={new_id} da ton tai.")
            return None

    emb_path = os.path.join(EMB_DIR, f"person_{new_id}.npy")
    np.save(emb_path, embed.astype(np.float32))

    ds.append({
        "person_id": new_id,
        "ho_ten": ho_ten,
        "ma_nv": ma_nv,
        "bo_phan": bo_phan,
        "ngay_sinh": ngay_sinh,
        "emb_file": emb_path,
        "embed": embed.astype(np.float32),
    })

    ghi_lai_csv(ds)
    return int(new_id)


def sua_thong_tin_csv(person_id: int):
    """
    Cho phép sửa cả person_id.
    Nếu đổi person_id thì rename file embedding theo id mới.
    """
    ds = tai_tat_ca_csv()
    p = None

    for x in ds:
        if int(x["person_id"]) == int(person_id):
            p = x
            break

    if p is None:
        print(f"[SUA] Khong tim thay person_id={person_id}")
        return False

    print("\n=== SUA THONG TIN (bo trong = giu nguyen) ===")
    print(f"person_id: {p['person_id']}")
    print(f"ho_ten: {p['ho_ten']}")
    print(f"ma_nv: {p['ma_nv']}")
    print(f"bo_phan: {p['bo_phan']}")
    print(f"ngay_sinh: {p['ngay_sinh']}")

    person_id_moi = input("person_id moi: ").strip()
    ho_ten = input("Ho va ten moi: ").strip()
    ma_nv = input("Ma NV moi: ").strip()
    bo_phan = input("Bo phan moi: ").strip()
    ngay_sinh = input("Ngay sinh moi (YYYY-MM-DD): ").strip()

    # ===== đổi person_id nếu có =====
    if person_id_moi:
        person_id_moi = int(person_id_moi)

        for x in ds:
            if int(x["person_id"]) == person_id_moi and int(x["person_id"]) != int(person_id):
                print(f"[SUA] person_id={person_id_moi} da ton tai.")
                return False

        old_id = int(p["person_id"])
        old_emb = (p.get("emb_file") or "").strip()

        if old_emb and os.path.exists(old_emb):
            new_emb = os.path.join(EMB_DIR, f"person_{person_id_moi}.npy")

            # nếu file đích đã tồn tại thì báo lỗi
            if os.path.exists(new_emb) and old_emb != new_emb:
                print(f"[SUA] File embedding dich da ton tai: {new_emb}")
                return False

            try:
                os.rename(old_emb, new_emb)
                p["emb_file"] = new_emb
            except Exception as ex:
                print("[SUA] Loi doi ten file embedding:", ex)
                return False

        p["person_id"] = person_id_moi

    # ===== sửa thông tin khác =====
    if ho_ten:
        p["ho_ten"] = ho_ten
    if ma_nv:
        p["ma_nv"] = ma_nv
    if bo_phan:
        p["bo_phan"] = bo_phan
    if ngay_sinh:
        p["ngay_sinh"] = ngay_sinh

    ghi_lai_csv(ds)
    print("[SUA] Da cap nhat CSV.\n")
    return True


def reindex_person_ids(ds):
    """
    Đánh lại person_id liên tục 1..N và rename embedding theo id mới.
    """
    ds = sorted(ds, key=lambda p: int(p["person_id"]))
    mapping = {int(p["person_id"]): i for i, p in enumerate(ds, start=1)}

    tmp_map = {}

    # rename qua tmp để tránh đè file
    for p in ds:
        old_id = int(p["person_id"])
        old_path = (p.get("emb_file") or "").strip()

        if old_path and os.path.exists(old_path):
            tmp_path = os.path.join(EMB_DIR, f".tmp_{old_id}_{int(time.time() * 1000)}.npy")
            try:
                os.rename(old_path, tmp_path)
                tmp_map[old_id] = tmp_path
            except Exception as ex:
                print("[REINDEX] Loi rename tmp:", ex)

    # tmp -> final
    for p in ds:
        old_id = int(p["person_id"])
        new_id = mapping[old_id]

        if old_id in tmp_map:
            final_path = os.path.join(EMB_DIR, f"person_{new_id}.npy")
            try:
                os.rename(tmp_map[old_id], final_path)
                p["emb_file"] = final_path
            except Exception as ex:
                print("[REINDEX] Loi rename final:", ex)

        p["person_id"] = new_id

    return ds


def xoa_person_va_reindex(person_id_can_xoa: int):
    ds = tai_tat_ca_csv()
    target = None

    for p in ds:
        if int(p["person_id"]) == int(person_id_can_xoa):
            target = p
            break

    if target is None:
        print(f"[XOA] Khong tim thay person_id={person_id_can_xoa}")
        return False

    emb_path = (target.get("emb_file") or "").strip()
    if emb_path and os.path.exists(emb_path):
        try:
            os.remove(emb_path)
        except Exception as ex:
            print("[XOA] Khong xoa duoc emb:", ex)

    ds = [p for p in ds if int(p["person_id"]) != int(person_id_can_xoa)]
    ds = reindex_person_ids(ds)
    ghi_lai_csv(ds)

    print(f"[XOA] Da xoa person_id={person_id_can_xoa} va danh lai ID 1..{len(ds)}")
    return True