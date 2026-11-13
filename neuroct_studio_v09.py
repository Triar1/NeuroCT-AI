"""NeuroCT AI Studio V0.9. Standalone research-only Streamlit showcase.
No changes to existing checkpoints, dataset or app.py.
"""
import io
import json
import html
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
LABELS = ['EDH Class', 'Healthy Class', 'IPH Class', 'IVH Class', 'SAH Class', 'SDH Class']
SHORT = ['EDH', 'Healthy', 'IPH', 'IVH', 'SAH', 'SDH']
FULL = ['Épidural', 'Sans hémorragie', 'Intraparenchymateux', 'Intraventriculaire', 'Sous-arachnoïdien', 'Sous-dural']
PATHS = ['models/neuroct_resnet18.pt', 'models/neuroct_resnet18_v04.pt', 'models/neuroct_efficientnet_b0.pt']
NAMES = ['ResNet18 · V0.3', 'ResNet18 · V0.4', 'EfficientNet-B0 · V0.6']
KINDS = ['resnet', 'resnet', 'efficient']
COLORS = ['#4adccf', '#9ab0ff', '#ffbc7a']
st.set_page_config(page_title='EvoScan | NeuroCT AI Studio', page_icon='🧠', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
:root{color-scheme:dark}
.stApp{background:radial-gradient(ellipse at 90% -10%,#17324a 0%,#0b1726 37%,#07111d 78%);color:#e9f4fa}
.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1450px}
header[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#0c1b2b}
h1,h2,h3{color:#f4faff!important;letter-spacing:-.5px}
h2{font-size:1.45rem!important} h3{font-size:1.12rem!important}
[data-testid="stMetric"]{background:#112638;border:1px solid #244155;border-radius:13px;padding:13px 17px}
[data-testid="stMetricLabel"]{color:#a7bfce} [data-testid="stMetricValue"]{font-size:1.65rem;color:#f0fbff}
[data-testid="stFileUploader"]{background:#112436;border:1px dashed #315568;border-radius:13px;padding:8px}
[data-testid="stImage"] img{border-radius:10px}
[data-testid="stAlert"]{border-radius:11px}
.stButton>button[kind="primary"],.stDownloadButton>button{background:#2dd3c4;color:#062029;border:0;font-weight:750;border-radius:10px}
.stButton>button{border-radius:10px}
[data-testid="stHorizontalBlock"]{align-items:stretch}
.topbar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #254156;padding:7px 0 18px;margin-bottom:22px;gap:14px}
.brand{font-size:20px;font-weight:850;letter-spacing:1.3px;color:#f5fcff}.brand span{color:#41e0d0}
.tag{border:1px solid #37616d;color:#74eee1;background:#12353e;border-radius:50px;padding:6px 12px;font-size:11px;letter-spacing:1.1px;font-weight:800}
.eyebrow{color:#45decd;letter-spacing:2.5px;font-size:11px;font-weight:800}
.hero{padding:24px 29px;border:1px solid #25465b;border-radius:19px;background:linear-gradient(110deg,#163247 0%,#0c2032 64%,#0b1a2b 100%);margin-bottom:15px}
.hero h1{font-size:38px!important;margin:7px 0!important}.hero h1 span{color:#49e0d0}
.hero p{color:#b4c9d6;font-size:15px;margin:8px 0 0}
.notice{border:1px solid #5b4b34;background:#2a271f;border-radius:10px;padding:9px 14px;color:#f4d9a4;font-size:12px;line-height:1.6;margin:9px 0 18px}
.panel-title{font-size:14px;color:#eaf5fc;font-weight:750;margin:7px 0 12px;letter-spacing:.3px}
.panel-sub{color:#93adbd;font-size:12px}
.kpi-caption{font-size:12px;color:#8faab9;margin:0 0 8px}
.barline{display:flex;align-items:center;gap:10px;margin:10px 0}.barname{width:65px;color:#c7dce8;font-size:12px}.bartrack{height:10px;flex:1;background:#213b4c;border-radius:9px;overflow:hidden}.barfill{height:100%;background:linear-gradient(90deg,#178c96,#52e2cd);border-radius:9px}.barvalue{width:50px;text-align:right;font-size:12px;color:#e8faff}
.cardrow{background:#112638;border:1px solid #294659;border-radius:13px;padding:16px 18px;margin:8px 0}
.cardrow .small{font-size:11px;color:#96b2c2}.cardrow .big{font-size:22px;color:#f2fbff;font-weight:800;margin-top:5px}
.sectionlabel{color:#58e5d4;font-size:11px;font-weight:850;letter-spacing:1.9px;margin:14px 0 8px}
.footnote{color:#9fb6c5;font-size:12px;line-height:1.6}
</style>''', unsafe_allow_html=True)

def read_json(path):
    try: return json.loads((ROOT/path).read_text(encoding='utf-8'))
    except (OSError,ValueError): return {}

@st.cache_resource(show_spinner='Chargement des modèles…')
def load_net(path,kind,stamp):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('classes') != LABELS: raise ValueError('Ordre des classes incompatible avec le modèle')
    if kind=='efficient':
        net=models.efficientnet_b0(weights=None)
        net.classifier[1]=nn.Linear(net.classifier[1].in_features,6)
    else:
        net=models.resnet18(weights=None)
        net.fc=nn.Linear(net.fc.in_features,6)
    net.load_state_dict(ck['state_dict'],strict=True)
    net.eval()
    return net

@st.cache_data(show_spinner=False)
def preprocess(data):
    im=Image.open(io.BytesIO(data)).convert('RGB')
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return im,tf(im).unsqueeze(0)

@st.cache_data(show_spinner=False)
def predict(data,model_stamps):
    im,x=preprocess(data)
    outputs=[]
    for path,kind,stamp in zip(PATHS,KINDS,model_stamps):
        if stamp is None: outputs.append(None);continue
        model=load_net(str(ROOT/path),kind,stamp)
        with torch.inference_mode():
            outputs.append(torch.softmax(model(x),dim=1)[0].numpy())
    return outputs

def gradcam(model,x,cls):
    """Attention from a ResNet18 layer, not lesion localization."""
    features=[];grads=[]
    def hook(module,inp,out):
        features.append(out)
        out.register_hook(lambda grad: grads.append(grad))
    handle=model.layer4.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(): model(x)[0,cls].backward()
        feat=features[0][0].detach(); grad=grads[0][0].detach()
        cam=torch.relu((feat*grad.mean((1,2),keepdim=True)).sum(0)).numpy()
        cam-=cam.min();cam/=max(float(cam.max()),1e-8)
        return np.asarray(Image.fromarray(np.uint8(cam*255)).resize((224,224),Image.Resampling.BILINEAR))/255
    finally:handle.remove()

def attention_overlay(im,cam,opacity=.55):
    import matplotlib
    rgba=matplotlib.colormaps['turbo'](cam)
    rgb=np.uint8(rgba[:,:,:3]*255)
    original=im.resize((224,224)).convert('RGB')
    overlay=Image.blend(original,Image.fromarray(rgb),float(opacity))
    return overlay

def score_bars(values):
    bars=''
    for i in np.argsort(values)[::-1]:
        v=float(values[i]); w=max(0,min(100,v*100))
        bars+=f'<div class="barline"><span class="barname">{SHORT[i]}</span><div class="bartrack"><div class="barfill" style="width:{w:.2f}%"></div></div><span class="barvalue">{v:.1%}</span></div>'
    st.markdown(bars,unsafe_allow_html=True)

def metric_fmt(obj,key):
    return f'{obj[key]:.1%}' if isinstance(obj,dict) and isinstance(obj.get(key),(float,int)) else '—'

metrics=read_json('models/ensemble_v06.json')
audit=read_json('models/leakage_v07.json')
weights=metrics.get('selected_weights',[.36,.24,.4])
try: weights=np.array([float(x) for x in weights],dtype=float)
except (ValueError,TypeError): weights=np.array([.36,.24,.4])
if len(weights)!=3 or np.any(weights<0) or abs(weights.sum()-1)>.01: weights=np.array([.36,.24,.4])
test=metrics.get('test',{}).get('ensemble_v06',{})

st.markdown('<div class="topbar"><div class="brand">✦ EVO<span>SCAN</span> <span style="color:#7794a8;font-size:12px;letter-spacing:1px">/ NEUROCT AI STUDIO</span></div><div class="tag">● RESEARCH PROTOTYPE · V0.9</div></div>',unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">MOBILE NEURO-IMAGING · AI EXPLORATION</div><h1>NeuroCT <span>AI Studio</span></h1><p>La brique d’exploration IA d’EvoScan · Analyse de coupes cérébrales · Trois réseaux, une sortie combinée</p></div>',unsafe_allow_html=True)
st.markdown('<div class="notice">⚠️ <b>Démo de recherche uniquement.</b> PNG/JPG, sans validation clinique. Ne pas utiliser pour diagnostiquer, trier ou prendre en charge un patient. Les scores ne sont pas des probabilités médicales. Les résultats de test sont potentiellement biaisés par des doublons.</div>',unsafe_allow_html=True)

nav=st.radio('Espace', ['🔬  Analyse IA', '🧠  Model Insights', '🛡️  Data Integrity', '🎤  Présentation jury'],horizontal=True,label_visibility='collapsed')
st.write('')

if nav.startswith('🔬'):
    st.markdown('<div class="sectionlabel">01 / IMAGE INTELLIGENCE</div>',unsafe_allow_html=True)
    st.subheader('Exploration d’une coupe cérébrale')
    st.caption('Charge une image anonymisée, ou sélectionne un exemple local. L’analyse s’effectue uniquement sur ton ordinateur.')
    left,right=st.columns([1,1],gap='large')
    with left:
        source=st.radio('Source image',['Importer une image','Exemple du dataset local'],horizontal=True)
        file_bytes=None;filename=None;true_label=None
        if source=='Importer une image':
            uploaded=st.file_uploader('PNG / JPG anonymisé',type=['png','jpg','jpeg'])
            if uploaded is not None: file_bytes=uploaded.getvalue();filename=uploaded.name
        else:
            split=st.selectbox('Ensemble local',['test','validation'],index=0)
            cls=st.selectbox('Catégorie de l’exemple',LABELS,index=1)
            folder=ROOT/'data'/'raw'/split/cls
            candidates=sorted(p for p in folder.glob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg'))[:30] if folder.exists() else []
            if candidates:
                idx=st.slider('Image',0,len(candidates)-1,0)
                chosen=candidates[idx]
                file_bytes=chosen.read_bytes();filename=chosen.name;true_label=cls.replace(' Class','')
                st.caption('Étiquette issue du dataset, non vérifiée indépendamment.')
            else: st.info('Aucune image trouvée dans ce dossier local.')
        if file_bytes:
            try:
                im,x=preprocess(file_bytes)
                bright=st.slider('Luminosité d’affichage',0.5,1.8,1.0,0.1)
                contrast=st.slider('Contraste d’affichage',0.5,2.0,1.0,0.1)
                disp=ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(bright)).enhance(contrast)
                st.image(disp,caption=f'IMAGE SOURCE · {filename} · Ajustements visuels uniquement',use_container_width=True)
            except Exception as exc: st.error(f'Image non lisible : {exc}');file_bytes=None
        else:
            st.info('Sélectionne une image pour activer l’analyse IA.')
    with right:
        st.markdown('<div class="panel-title">ENSEMBLE INFERENCE</div>',unsafe_allow_html=True)
        if file_bytes:
            stamps=[(ROOT/p).stat().st_mtime if (ROOT/p).exists() else None for p in PATHS]
            if not all(s is not None for s in stamps):
                st.error('Checkpoint(s) manquant(s) : '+', '.join(p for p,s in zip(PATHS,stamps) if s is None))
                st.caption('La V0.8 exige les trois modèles pour conserver les pondérations et éviter des résultats trompeurs.')
            else:
                try:
                    probs=predict(file_bytes,tuple(stamps))
                    combined=sum(p*w for p,w in zip(probs,weights))
                    best=int(np.argmax(combined))
                    st.metric('Classe dominante du modèle',SHORT[best])
                    st.caption(FULL[best]+' · Score brut non calibré : '+f'{combined[best]:.1%}')
                    score_bars(combined)
                    if true_label:
                        st.info('Référence dataset : '+true_label+' | Prédiction IA : '+SHORT[best]+' | '+('Correspondance apparente' if true_label==SHORT[best] else 'Désaccord apparent')+' (annotations non vérifiées)')
                    else:
                        st.caption('Référence dataset : inconnue pour une image importée')
                    st.markdown('<div class="panel-sub">Les six scores sont des sorties algorithmiques, pas des probabilités cliniques.</div>',unsafe_allow_html=True)
                    st.divider()
                    st.markdown('**Consensus des réseaux**')
                    votes=[int(np.argmax(p)) for p in probs]
                    n_agree=sum(v==best for v in votes)
                    st.progress(n_agree/3,text=f'{n_agree}/3 réseaux prédisent la classe de l’ensemble')
                    if n_agree<3:st.caption('Les réseaux ne sont pas unanimes : désaccord algorithmique, pas mesure d’incertitude calibrée.')
                except Exception as exc:
                    st.error(f'Inférence impossible : {type(exc).__name__}: {exc}')
                    probs=None
        else: st.markdown('<div class="cardrow"><div class="small">STATUT</div><div class="big">En attente d’image</div><div class="small">L’analyse démarre dès qu’une coupe est sélectionnée.</div></div>',unsafe_allow_html=True)
    if file_bytes and 'probs' in locals() and probs is not None:
        st.divider()
        st.markdown('<div class="sectionlabel">02 / EXPLAINABILITY</div>',unsafe_allow_html=True)
        st.subheader('Image source & carte d’attention')
        show_cam=st.toggle('Calculer Grad-CAM (ResNet18 V0.3)',value=False)
        if show_cam:
            opacity=st.slider('Opacité de la carte d’attention',0.10,0.85,0.45,0.05,help='Réglage visuel uniquement ; ne modifie pas le calcul du modèle.')
            try:
                cam=gradcam(load_net(str(ROOT/PATHS[0]),'resnet',stamps[0]),x,best)
                a,b=st.columns(2,gap='large')
                with a:st.image(im.resize((224,224)),caption='Image originale',use_container_width=True)
                with b:st.image(attention_overlay(im,cam,opacity=opacity),caption='Grad-CAM · zones d’attention du réseau',use_container_width=True)
                st.caption('La carte indique les régions influençant une sortie du ResNet18. Ce n’est ni une segmentation ni une localisation validée de lésion. Elle concerne le ResNet18, pas directement l’ensemble.')
            except Exception as exc:st.warning(f'Grad-CAM indisponible : {type(exc).__name__}: {exc}')
        st.markdown('<div class="sectionlabel">03 / MODEL COMPARISON</div>',unsafe_allow_html=True)
        cols=st.columns(3,gap='medium')
        for col,name,p,w,color in zip(cols,NAMES,probs,weights,COLORS):
            with col:
                st.markdown(f'<div class="cardrow"><div class="small">{html.escape(name)} · POIDS {w:.0%}</div><div class="big" style="color:{color}">{SHORT[int(p.argmax())]}</div><div class="small">Score maximal : {float(p.max()):.1%}</div></div>',unsafe_allow_html=True)
                with st.expander('Voir les 6 scores'):score_bars(p)
        report={'application':'NeuroCT AI Studio V0.9','usage':'research_demo_only','filename':filename,'dataset_label_unverified':true_label,'predicted_class':SHORT[best],'uncalibrated_scores':{k:round(float(v),5) for k,v in zip(SHORT,combined)},'ensemble_weights':{k:round(float(w),4) for k,w in zip(NAMES,weights)},'model_predictions':{k:SHORT[int(p.argmax())] for k,p in zip(NAMES,probs)},'clinical_validation':False,'data_leakage_warning':'Original dataset splits contain exact duplicates; test metrics may be biased.'}
        st.download_button('↓ Exporter le rapport de recherche JSON',json.dumps(report,ensure_ascii=False,indent=2),file_name='neuroct_research_demo.json',mime='application/json')

elif nav.startswith('🧠'):
    st.markdown('<div class="sectionlabel">MODEL INSIGHTS / THREE-NET ENSEMBLE</div>',unsafe_allow_html=True)
    st.subheader('Trois modèles, une prédiction combinée')
    st.caption('Deux ResNet18 et un EfficientNet-B0. Les coefficients proviennent de la validation, pas du test.')
    cols=st.columns(3,gap='medium')
    for col,name,w,key,color in zip(cols,NAMES,weights,['v03','v04','efficientnet'],COLORS):
        val=metrics.get('test',{}).get(key,{})
        with col:
            st.markdown(f'<div class="cardrow"><div class="small">{html.escape(name)}</div><div class="big" style="color:{color}">{w:.0%}</div><div class="small">POIDS DANS LA FUSION</div></div>',unsafe_allow_html=True)
            st.metric('Accuracy test exploratoire',metric_fmt(val,'accuracy'))
            st.metric('Balanced accuracy',metric_fmt(val,'balanced_accuracy'))
    st.markdown('### Fusion pondérée')
    st.latex(r'P_{ensemble}=0.36\,P_{ResNet03}+0.24\,P_{ResNet04}+0.40\,P_{EfficientNet}')
    st.caption('Les poids effectifs chargés depuis ensemble_v06.json sont : '+', '.join(f'{w:.0%}' for w in weights)+'.')
    st.markdown('### Évaluation exploratoire')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Accuracy · 6 classes',metric_fmt(test,'accuracy'))
    c2.metric('Balanced accuracy',metric_fmt(test,'balanced_accuracy'))
    c3.metric('Sensibilité binaire',metric_fmt(test,'sensitivity'))
    c4.metric('Spécificité binaire',metric_fmt(test,'specificity'))
    st.warning('Ces scores proviennent du jeu de test initial comportant des chevauchements avec les autres ensembles. Ils ne mesurent pas une performance indépendante par patient.')

elif nav.startswith('🛡️'):
    st.markdown('<div class="sectionlabel">DATA INTEGRITY / AUTOMATED AUDIT</div>',unsafe_allow_html=True)
    st.subheader('Qualité des données : ce que révèle l’audit')
    if not audit:
        st.info('Lance `python audit_leakage_v07.py` dans le dossier du projet pour générer `models/leakage_v07.json`.')
    else:
        c1,c2,c3=st.columns(3)
        c1.metric('Groupes dupliqués entre splits',f"{audit.get('cross_split_duplicate_groups',0):,}".replace(',',' '))
        fraction=audit.get('affected_test_image_fraction')
        c2.metric('Images du test concernées',f'{fraction:.1%}' if isinstance(fraction,(int,float)) else '—')
        c3.metric('Groupes avec étiquettes divergentes',f"{audit.get('conflicting_label_groups',0):,}".replace(',',' '))
        st.markdown('### Chevauchement entre les ensembles')
        pairs=audit.get('cross_split_groups_by_pair',{})
        titles=[('train__validation','Entraînement ↔ Validation'),('train__test','Entraînement ↔ Test'),('validation__test','Validation ↔ Test')]
        maxv=max([int(pairs.get(k,0)) for k,_ in titles]+[1])
        for k,title in titles:
            v=int(pairs.get(k,0))
            st.markdown(f'<div class="barline"><span class="barname" style="width:215px">{title}</span><div class="bartrack"><div class="barfill" style="width:{v/maxv*100:.1f}%"></div></div><span class="barvalue">{v:,}</span></div>',unsafe_allow_html=True)
        st.caption('Nombre de groupes d’images identiques par paire de splits. Un même groupe peut apparaître dans plusieurs paires : ne pas additionner ces chiffres.')
        st.markdown('### Composition du dataset')
        counts=audit.get('total_images',{})
        if counts:
            cs=st.columns(3)
            for col,key in zip(cs,['train','validation','test']):col.metric(key.capitalize(),f"{counts.get(key,0):,}".replace(',',' '))
        st.error('Fuite de données potentielle et étiquettes contradictoires : impossible de considérer les métriques actuelles comme une évaluation indépendante. Patient-level split non vérifié.')
        with st.expander('Méthode et limites de l’audit'):
            st.write(audit.get('method','Comparaison des pixels décodés'))
            st.write('Les conflits indiquent des labels de dossiers différents pour des pixels identiques ; ils ne prouvent pas à eux seuls une erreur médicale. Le hash ne détecte pas toutes les images presque identiques.')
            st.json({k:v for k,v in audit.items() if k not in ['cross_split_examples','conflict_examples']})
        with st.expander('Exemples de conflits (chemins locaux)'):
            st.json(audit.get('conflict_examples',[]))

else:
    st.markdown('<div class="sectionlabel">PITCH MODE / HACKATHON</div>',unsafe_allow_html=True)
    st.subheader('EvoScan : de l’imagerie mobile à l’exploration IA')
    st.markdown('**01 · Le besoin** — Explorer l’intérêt d’une chaîne d’analyse IA associée à un concept de scanner cérébral compact et mobile.')
    st.markdown('**02 · Notre prototype** — Trois réseaux neuronaux analysent une coupe PNG, puis une fusion pondérée combine leurs scores. Grad-CAM illustre l’attention d’un des réseaux.')
    st.markdown('**03 · Notre démarche scientifique** — Un audit automatique a mis en évidence des doublons et des étiquettes divergentes entre les ensembles du dataset.')
    st.markdown('**04 · La suite** — Construire un dataset propre, séparé par patient, réentraîner et faire une validation externe avant toute perspective clinique.')
    st.info('Démo conseillée : Analyse IA → image d’exemple → scores → Grad-CAM → Data Integrity. Durée cible : 2 à 3 minutes.')
    st.markdown('### Chiffres du prototype')
    a,b,c=st.columns(3)
    a.metric('Réseaux entraînés','3')
    b.metric('Accuracy exploratoire',metric_fmt(test,'accuracy'))
    c.metric('Test avec doublons',f"{audit.get('affected_test_image_fraction',0):.1%}" if audit else 'Audit requis')

st.divider()
st.markdown('<div class="footnote">EVOSCAN / NEUROCT AI STUDIO · Prototype de recherche hors usage clinique · Exécution locale · Aucune image transmise à un service distant par ce script.</div>',unsafe_allow_html=True)
