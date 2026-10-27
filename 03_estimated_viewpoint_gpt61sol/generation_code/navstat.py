import json,glob,math,os,statistics as S
REF=json.load(open('outputs/hy_autonomy_200_20261002/evaluation_only/reference_geometry.json'))
scale={os.path.basename(p)[:-5]:json.load(open(p)) for p in glob.glob('outputs/hy_nav_pilot_20261002/scale/*.json')}
wrap=lambda a:(a+180)%360-180
rows=[]
for p in glob.glob('outputs/hy_nav_pilot_20261002/v5/*/*/plan.json'):
    d=json.load(open(p,encoding='utf-8')); q=d['question_id']; sid=q.rsplit('-',1)[0]; r=REF[q]
    o=r['points_m'][-1][:2]; f=d['final']; mpu=d['m_per_unit']
    rows.append(dict(sid=sid,split=r['split'],o=o,unit=[f['x']/mpu,f['y']/mpu],mpu=mpu,moge=scale[sid]['m_per_unit_moge'],
        yaw=abs(wrap(f['yaw_deg']-math.degrees(r['final_yaw']))),text=d['motion_text'],
        olen=sum(math.dist(a[:2],b[:2]) for a,b in zip(r['points_m'],r['points_m'][1:]))))
def fitk(sub):
    n=sum((r['unit'][0]*r['o'][0]+r['unit'][1]*r['o'][1])*r['moge'] for r in sub); d=sum((r['unit'][0]**2+r['unit'][1]**2)*r['moge']**2 for r in sub); return n/d
ks={s:fitk([x for x in rows if x['sid']!=s]) for s in {r['sid'] for r in rows}}
for r in rows:
    r['emoge']=math.dist([v*r['moge'] for v in r['unit']],r['o'])
    r['e175']=math.dist([v*r['mpu'] for v in r['unit']],r['o'])
    r['eloso']=math.dist([v*r['moge']*ks[r['sid']] for v in r['unit']],r['o'])
def summ(sub,label):
    print(label,'n=',len(sub),'scenes',len({r['sid'] for r in sub}),'oracle route len med %.2f'%S.median(r['olen'] for r in sub))
    for k in ['emoge','e175','eloso']:
        e=[r[k] for r in sub]; print('  %s med %.2f mean %.2f <1m %.0f%% <2m %.0f%%'%(k,S.median(e),S.mean(e),100*sum(x<1 for x in e)/len(e),100*sum(x<2 for x in e)/len(e)))
    y=[r['yaw'] for r in sub]; print('  yaw med %.1f mean %.1f <15 %.0f%% <30 %.0f%% >90 %.0f%%'%(S.median(y),S.mean(y),100*sum(x<15 for x in y)/len(y),100*sum(x<30 for x in y)/len(y),100*sum(x>90 for x in y)/len(y)))
summ(rows,'ALL')
for sp in ['development','holdout']: summ([r for r in rows if r['split']==sp],sp)
u={}
for r in rows: u.setdefault((r['sid'],r['text']),r)
summ(list(u.values()),'unique routes')
print('global k',round(fitk(rows),3))
