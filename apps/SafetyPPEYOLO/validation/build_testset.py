"""Build shared GT test set (v2): 40 images, balanced toward vest classes."""
import os, json, shutil, zipfile, glob
from collections import Counter
from huggingface_hub import hf_hub_download

SCRATCH = "/private/tmp/claude-501/-Users-ajayshah-Desktop-ZETIC-ZETIC-Melange-apps/6abc40ad-72cf-4e76-bb63-6290bbef3255/scratchpad"
OUT = f"{SCRATCH}/ppe_testset"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

CANON = {"hardhat": "helmet", "no-hardhat": "no-helmet", "safety vest": "vest",
         "no-safety vest": "no-vest", "person": "person"}

def extract(split):
    p = hf_hub_download(repo_id="keremberke/construction-safety-object-detection",
                        filename=f"data/{split}.zip", repo_type="dataset")
    d = f"{SCRATCH}/ppe_raw_{split}"
    if not os.path.exists(f"{d}/_annotations.coco.json"):
        os.makedirs(d, exist_ok=True)
        zipfile.ZipFile(p).extractall(d)
    return d

pool = []  # (has_vest, n_canon, dir, im, boxes, split)
for split in ["test", "valid", "train"]:
    d = extract(split)
    coco = json.load(open(f"{d}/_annotations.coco.json"))
    cats = {c["id"]: c["name"].lower() for c in coco["categories"]}
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in coco["images"]:
        anns = by_img.get(im["id"], [])
        names = [cats[a["category_id"]] for a in anns]
        if not any(n in names for n in ("hardhat", "safety vest", "no-hardhat", "no-safety vest")):
            continue
        boxes = []
        for a in anns:
            cn = cats[a["category_id"]]
            if cn in CANON:
                x, y, w, h = a["bbox"]
                boxes.append({"cls": CANON[cn], "xyxy": [x, y, x + w, y + h]})
        if len(boxes) < 2:
            continue
        has_vest = any(b["cls"] in ("vest", "no-vest") for b in boxes)
        pool.append((has_vest, len(boxes), d, im, boxes, split))

print("pool size:", len(pool), " with vest:", sum(1 for p in pool if p[0]))

# take: all test/valid images first (indep. of vest), then top-up with vest-heavy train images
pool_tv = [p for p in pool if p[5] in ("test", "valid")]
pool_tr = [p for p in pool if p[5] == "train"]
pool_tr.sort(key=lambda p: (not p[0], -p[1]))  # vest-containing, box-rich first

chosen = pool_tv[:24] + pool_tr[: 40 - min(24, len(pool_tv))]
records = []
for count, (hv, nb, d, im, boxes, split) in enumerate(chosen):
    fn = f"img_{count:03d}.jpg"
    shutil.copy(f"{d}/{im['file_name']}", f"{OUT}/{fn}")
    records.append({"file": fn, "w": im["width"], "h": im["height"], "boxes": boxes,
                    "src": f"{split}/{im['file_name']}"})

json.dump(records, open(f"{OUT}/ground_truth.json", "w"), indent=1)
cc = Counter(b["cls"] for r in records for b in r["boxes"])
print(f"saved {len(records)} images -> {OUT}")
print("GT box counts:", dict(cc))
