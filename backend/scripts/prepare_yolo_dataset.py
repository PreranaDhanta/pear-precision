import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ds = ROOT/'datasets'/'data'/'pear640'  # Corrected path to match Ultralytics default dataset directory
images = ds/'images'
labels = ds/'labels'
assert images.exists() and labels.exists(), "Pear640 not found with images/ and labels/"

all_imgs = []
for ext in ("*.jpg","*.jpeg","*.png"):
    all_imgs += list(images.rglob(ext))

random.seed(42)
random.shuffle(all_imgs)

n = len(all_imgs)
train, val, test = all_imgs[:int(0.8*n)], all_imgs[int(0.8*n):int(0.95*n)], all_imgs[int(0.95*n):]

def rel(p): return str(p.relative_to(ROOT).as_posix())
def write_list(lst, fp):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, 'w') as f: f.write("\n".join(rel(p) for p in lst))

write_list(train, ROOT/'datasets'/'data'/'splits'/'train.txt')
write_list(val,   ROOT/'datasets'/'data'/'splits'/'val.txt')
write_list(test,  ROOT/'datasets'/'data'/'splits'/'test.txt')

yaml = f"""# auto-generated
path: {rel(ds)}
train: {rel(ROOT/'datasets'/'data'/'splits'/'train.txt')}
val:   {rel(ROOT/'datasets'/'data'/'splits'/'val.txt')}
test:  {rel(ROOT/'datasets'/'data'/'splits'/'test.txt')}
names: [pear]
"""
(ROOT/'datasets'/'data'/'pear640.yaml').write_text(yaml)
print("Wrote datasets/data/pear640.yaml and split files.")
