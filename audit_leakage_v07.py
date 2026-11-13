"""Read-only exact duplicate leakage audit. No images are moved or deleted."""
import argparse, hashlib, json, csv
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/raw');p.add_argument('--output',default='models/leakage_v07.json');a=p.parse_args()
    root=Path(a.data); groups=defaultdict(list); bad=[]; counts=Counter()
    for split in ('train','validation','test'):
        for f in sorted((root/split).glob('*/*')):
            if f.suffix.lower() not in ('.png','.jpg','.jpeg'):continue
            label=f.parent.name;counts[split]+=1
            try:
                # Hash decoded pixels: also detects identical images with different file encoding.
                with Image.open(f) as im:
                    im=im.convert('L'); h=hashlib.sha256(im.size[0].to_bytes(4,'little')+im.size[1].to_bytes(4,'little')+im.tobytes()).hexdigest()
                groups[h].append({'split':split,'class':label,'path':str(f)})
            except Exception as e:bad.append({'path':str(f),'error':str(e)})
        print(f'Indexed {split}: {counts[split]} images',flush=True)
    overlap=Counter();affected=Counter(); label_conflicts=[];examples=[];unique_cross=0
    for h,items in groups.items():
        splits={x['split'] for x in items}
        if len(splits)<2:continue
        unique_cross+=1
        for s in splits:affected[s]+=sum(x['split']==s for x in items)
        for s1,s2 in [('train','validation'),('train','test'),('validation','test')]:
            if s1 in splits and s2 in splits:overlap[f'{s1}__{s2}']+=1
        if len({x['class'] for x in items})>1:label_conflicts.append(items[:8])
        if len(examples)<12:examples.append(items[:6])
    report={'method':'SHA256 of grayscale decoded pixels plus image dimensions','total_images':dict(counts),'unique_pixel_hashes':len(groups),'cross_split_duplicate_groups':unique_cross,'cross_split_groups_by_pair':dict(overlap),'affected_images_by_split':dict(affected),'affected_test_image_fraction':affected['test']/counts['test'] if counts['test'] else None,'conflicting_label_groups':len(label_conflicts),'conflict_examples':label_conflicts[:8],'cross_split_examples':examples,'unreadable':bad,'patient_independence_verified':False,'warning':'Exploratory audit only. Even after removing exact duplicates, near-duplicates and patient-level overlap may remain. Existing trained models must NOT be treated as clean-split models.'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nCROSS-SPLIT PIXEL DUPLICATE GROUPS:',unique_cross)
    print('Pair overlaps:',dict(overlap))
    print('Affected images by split:',dict(affected))
    print('Affected TEST images:',f"{affected['test']}/{counts['test']} ({100*report['affected_test_image_fraction']:.1f}%)" if counts['test'] else 'N/A')
    print('Conflicting labels:',len(label_conflicts));print('Unreadable:',len(bad));print('Saved',out)
if __name__=='__main__':main()
