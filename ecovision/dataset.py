"""Audit YOLO detection labels and create a new, non-destructive split."""
import hashlib, json, math, shutil
from pathlib import Path
from collections import Counter
from PIL import Image
SPLITS=('train','val','test')
EXT={'.jpg','.jpeg','.png','.webp','.bmp'}

def audit(root, classes=4):
    root=Path(root); items=[]; errors=[]; seen={}; counts=Counter()
    for split in SPLITS:
        image_dir=root/'images'/split; label_dir=root/'labels'/split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f'{split}: missing image or label directory'); continue
        image_stems=set()
        for img in sorted(image_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in EXT: continue
            label=label_dir/(img.stem+'.txt')
            if img.stem in image_stems: errors.append(f'{img}: multiple image files share one label stem')
            image_stems.add(img.stem)
            if not label.is_file(): errors.append(f'{img}: missing label'); continue
            try:
                with Image.open(img) as im: im.verify()
                digest=hashlib.sha256(img.read_bytes()).hexdigest()
                if digest in seen: errors.append(f'Duplicate image content: {img} and {seen[digest]}')
                else: seen[digest]=str(img)
                labels=Counter()
                for i,line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(),1):
                    if not line.strip(): continue
                    parts=line.split()
                    if len(parts)!=5: raise ValueError(f'line {i}: expected 5 values')
                    cls=int(parts[0]); x,y,w,h=map(float,parts[1:])
                    if not 0<=cls<classes or not all(math.isfinite(v) for v in (x,y,w,h)) or w<=0 or h<=0 or x-w/2 < -1e-6 or x+w/2>1+1e-6 or y-h/2 < -1e-6 or y+h/2>1+1e-6:
                        raise ValueError(f'line {i}: invalid class or normalized box')
                    labels[cls]+=1
                counts.update(labels)
                items.append({'image':img,'label':label,'digest':digest,'dominant':str(labels.most_common(1)[0][0]) if labels else 'empty','split':split})
            except (OSError, ValueError) as exc: errors.append(f'{img}: {exc}')
        for label in label_dir.glob('*.txt'):
            if label.stem not in image_stems: errors.append(f'{label}: orphan label')
    if not items: errors.append('No valid image-label pairs found.')
    report={'images':len(items),'class_instances':dict(counts),'split_counts':dict(Counter(x['split'] for x in items)),'errors':errors}
    return items,report

def split_dataset(root, output, seed=42):
    from sklearn.model_selection import train_test_split
    root=Path(root).resolve(); output=Path(output).resolve()
    if output.exists() or root==output or root in output.parents or output in root.parents:
        raise ValueError('Choose a new output directory separate from the source dataset.')
    items,report=audit(root)
    if report['errors']: raise ValueError('Dataset audit failed: '+ '; '.join(report['errors'][:8]))
    train,rest=train_test_split(items,train_size=.7,stratify=[x['dominant'] for x in items],random_state=seed)
    val,test=train_test_split(rest,train_size=2/3,stratify=[x['dominant'] for x in rest],random_state=seed)
    output.mkdir(parents=True)
    try:
        for split,group in zip(SPLITS,(train,val,test)):
            for folder in ('images','labels'): (output/folder/split).mkdir(parents=True)
            for item in group:
                stem=item['digest']; img=item['image']; label=item['label']
                shutil.copy2(img,output/'images'/split/(stem+img.suffix.lower()))
                shutil.copy2(label,output/'labels'/split/(stem+'.txt'))
        manifest={'seed':seed,'method':'dominant-class stratified, not multilabel stratification','source_images':len(items),'split_counts':dict(zip(SPLITS,map(len,(train,val,test))))}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        import yaml
        (output/'data.yaml').write_text(yaml.safe_dump({'path':str(output),'train':'images/train','val':'images/val','test':'images/test','names':['CARDBOARD','METAL','PAPER','PLASTIC']}),encoding='utf-8')
        return manifest
    except BaseException:
        shutil.rmtree(output); raise
