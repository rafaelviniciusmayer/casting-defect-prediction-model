import pickle,time,json,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from compare_models import _optimize_thresholds_silent
from unified_cv_pipeline import logistic_penalty_kwargs
d=pickle.load(open('models/phase3_final_models.pkl','rb'))
Xd,yd,Xt,yt=d['X_dev_scaled'].astype(np.float64),d['y_dev'].astype(int),d['X_test_scaled'].astype(np.float64),d['y_test'].astype(int)
names=list(d['defect_names']); res=[]
for pen,solver,Cs in [('l1','liblinear',[0.01,0.05,0.1,0.5]),('l2','lbfgs',[0.01,0.1,1.0])]:
    for C in Cs:
        t0=time.time(); P=np.zeros(yt.shape); nz=0; tot=0
        for k in range(yd.shape[1]):
            if len(np.unique(yd[:,k]))<2: P[:,k]=float(yd[0,k]); continue
            clf=LogisticRegression(**logistic_penalty_kwargs(pen),C=C,class_weight='balanced',solver=solver,max_iter=2000,random_state=42).fit(Xd,yd[:,k])
            P[:,k]=clf.predict_proba(Xt)[:,1]; nz+=int((np.abs(clf.coef_)>1e-8).sum()); tot+=clf.coef_.size
        el=time.time()-t0
        _,pred=_optimize_thresholds_silent(yt,P,names)
        r=dict(penalty=pen,C=C,time=el,nonzero=nz,total=tot,recall=recall_score(yt,pred,average='micro'),precision=precision_score(yt,pred,average='micro'),f1=f1_score(yt,pred,average='micro'),f1_macro=f1_score(yt,pred,average='macro',zero_division=0))
        res.append(r); print(r,flush=True)
        json.dump(res,open('l1_sweep.json','w'))
