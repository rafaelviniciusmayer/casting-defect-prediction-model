import pickle, numpy as np, json, os
d=pickle.load(open('models/phase3_final_models.pkl','rb')); y=d['y_test'].ravel()
names=['pytorch_nn','xgboost','random_forest','logistic_regression_l2','logistic_regression_l1']
out={}
print('prevalence',y.mean())
for m in names:
    f=f'models/smote_proba/{m}.npy'
    if not os.path.exists(f): continue
    for cond,p in [('without',d['models'][m]['test_proba']),('with',np.load(f))]:
        p=p.ravel(); hi=p>=0.5
        out[(m,cond)]=(p.mean(), p[hi].mean(), y[hi].mean(), hi.sum())
        print(m,cond,'mean_pred=%.4f obs_prev=%.4f | p>=0.5: n=%d mean_pred=%.3f obs=%.3f gap=%.3f'%(p.mean(),y.mean(),hi.sum(),p[hi].mean(),y[hi].mean(),p[hi].mean()-y[hi].mean()))
