# -*- coding: utf-8 -*-
"""
第 2 章 数据库统计图（图 2-2 / 2-3 / 2-4）
=====================================================================
输出目录见 OUT；每张图同时导出 PNG (600 dpi) 与 PDF (矢量，投稿用)。
画布：Elsevier 双栏宽 190 mm。字体 Times New Roman，公式用 STIX。

图 2-2  Fig2-2_parameter_ranges     六个关键参数的分布直方图
图 2-3  Fig2-3_sample_composition   截面类别 / 边界条件 / 破坏模式的样本构成
图 2-4  Fig2-4_dsm_accuracy         DSM 预测精度：散点、三方案比较、随细长比的趋势

数据源：All_Dataset_cleaned1.csv（清洗后 1037 个试件）
运行：python dbfigs.py
"""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parent                     # 输出目录，按需修改
OUT.mkdir(exist_ok=True)
MM = 1/25.4; W2 = 190*MM
mpl.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
    'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'axes.titlesize':8,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7,
    'axes.linewidth':0.6,'lines.linewidth':1.0,
    'xtick.major.width':0.6,'ytick.major.width':0.6,
    'xtick.major.size':2.6,'ytick.major.size':2.6,
    'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
    'legend.frameon':False,'axes.grid':False,
    'savefig.dpi':600,'figure.dpi':140,'savefig.bbox':'tight','savefig.pad_inches':0.02,
    'pdf.fonttype':42,'ps.fonttype':42})

BLUE='#4C72B0'; ORNG='#DD8452'; GREEN='#009E73'; GREY='#7F7F7F'
FM_C = {'L':'#0072B2','L+F':'#56B4E9','F':'#009E73','D':'#E69F00','L+D':'#D55E00'}

DATA = 'D:\PyCharm\Py_Projects\pinn\cc.pro\cc_0902\All_Dataset_cleaned1.csv'    # 数据文件路径，按需修改
df = pd.read_csv(DATA, encoding='utf-8-sig')

def plab(ax,s,dx=-0.16,dy=1.03):
    ax.text(dx,dy,s,transform=ax.transAxes,fontsize=8.5,fontweight='bold',va='bottom',ha='left')

# ── 图 A：关键参数分布（6 联）────────────────────────────────────────
vars6 = [('t(mm)','Thickness $t$ (mm)',None),
         ('h(mm)','Web depth $h$ (mm)',None),
         ('L(mm)','Length $L$ (mm)',None),
         ('FS (mm)','Fastener spacing $FS$ (mm)',None),
         ('Fy(Mpa)','Yield strength $F_y$ (MPa)',None),
         ('Ptest(kN)','Test capacity $P_t$ (kN)',None)]
fig,axes = plt.subplots(2,3,figsize=(W2,99*MM))
for ax,(col,lab,_),tag in zip(axes.ravel(),vars6,'abcdef'):
    v = df[col].values
    ax.hist(v,bins=32,color=BLUE,alpha=.82,edgecolor='white',linewidth=.25)
    ax.set_xlabel(lab); ax.set_ylabel('Count')
    ax.text(.97,.94,f'n = {len(v)}\nmin {v.min():.4g}\nmax {v.max():.4g}\nmean {v.mean():.4g}',
            transform=ax.transAxes,ha='right',va='top',fontsize=6.4,linespacing=1.35)
    if col == 'FS (mm)':
        n0 = int((v == 0).sum())
        ax.text(.97,.50,f'{n0} specimens with\ncontinuous connection\n($FS$ = 0)',
                transform=ax.transAxes,ha='right',va='top',fontsize=6.2,
                color='#555555',linespacing=1.35)
    plab(ax,f'({tag})')
fig.tight_layout(w_pad=1.6,h_pad=2.0)
for e in ('png','pdf'): fig.savefig(OUT/f'Fig2-2_parameter_ranges.{e}')
plt.close(fig); print('Fig2-2 ok')

# ── 图 B：样本构成（3 联饼/条）──────────────────────────────────────
fig,axes = plt.subplots(1,3,figsize=(W2,58*MM))
def barcat(ax,series,order,colors,xlab,tag):
    cnt = series.value_counts().reindex(order)
    y = np.arange(len(order))[::-1]
    ax.barh(y,cnt.values,height=.62,color=colors,edgecolor='black',linewidth=.4)
    for yi,v in zip(y,cnt.values):
        ax.text(v+len(series)*0.012,yi,f'{v}  ({v/len(series)*100:.1f}%)',
                va='center',fontsize=6.8)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel(xlab); ax.set_xlim(0,cnt.max()*1.42)
    ax.tick_params(axis='y',length=0); plab(ax,f'({tag})',dx=-0.34)

barcat(axes[0],df['Sections_Category'],['Closed','Open','Half-Closed'],
       [BLUE,ORNG,GREEN],'Number of specimens','a')
barcat(axes[1],df['BC'],['Fixed','Pinned','Semi-Fixed'],
       [BLUE,ORNG,GREEN],'Number of specimens','b')
barcat(axes[2],df['FM_merged'],['L','L+F','F','D','L+D'],
       [FM_C[k] for k in ['L','L+F','F','D','L+D']],'Number of specimens','c')
axes[0].set_title('Section category',fontsize=8)
axes[1].set_title('Boundary condition',fontsize=8)
axes[2].set_title('Observed failure mode',fontsize=8)
fig.tight_layout(w_pad=3.0)
for e in ('png','pdf'): fig.savefig(OUT/f'Fig2-3_sample_composition.{e}')
plt.close(fig); print('Fig2-3 ok')

# ── 图 C：DSM 预测精度（3 联）───────────────────────────────────────
from sklearn.metrics import r2_score
fig,axes = plt.subplots(1,3,figsize=(W2,64*MM))
y = df['Ptest(kN)'].values
ax = axes[0]
for fm in ['L','L+F','F','D','L+D']:
    m = (df.FM_merged==fm).values
    ax.scatter(df['PDSM_s(kN)'].values[m],y[m],s=7,alpha=.55,
               color=FM_C[fm],edgecolors='none',label=fm)
lim = 800
ax.plot([0,lim],[0,lim],'k--',lw=.8)
ax.plot([0,lim/1.3],[0,lim],color=GREY,lw=.6,ls=':')
ax.plot([0,lim],[0,lim*0.7],color=GREY,lw=.6,ls=':')
ax.text(lim/1.3*0.80, lim*0.80+18,'+30%',fontsize=6.3,color=GREY,
        rotation=52,rotation_mode='anchor',ha='center',va='bottom')
ax.text(lim*0.80, lim*0.7*0.80-16,'−30%',fontsize=6.3,color=GREY,
        rotation=35,rotation_mode='anchor',ha='center',va='top')
ax.set_xlim(0,lim); ax.set_ylim(0,lim)
ax.set_xlabel(r'$P_{\mathrm{DSM},s}$ (kN)'); ax.set_ylabel(r'$P_t$ (kN)')
ax.legend(loc='upper left',ncol=2,handletextpad=.2,columnspacing=.8,markerscale=1.6)
ax.text(.97,.06,f'$R^2$ = {r2_score(y,df["PDSM_s(kN)"]):.3f}',transform=ax.transAxes,
        ha='right',fontsize=7.5)
plab(ax,'(a)')

ax = axes[1]
meth = [('PDSM_s(kN)','Single\nsection'),('PDSM_ms(kN)','Modified\nslenderness'),
        ('PDSM_b (kN)','Built-up\nsection')]
data = [ (y/df[c].clip(lower=1e-6)).values for c,_ in meth ]
bp = ax.boxplot(data,widths=.5,patch_artist=True,showfliers=True,
                medianprops=dict(color='white',lw=1.2),
                flierprops=dict(marker='o',ms=1.8,alpha=.35,markeredgecolor=GREY))
for p,c in zip(bp['boxes'],[BLUE,ORNG,GREEN]):
    p.set_facecolor(c); p.set_edgecolor('black'); p.set_linewidth(.5)
for e in ['whiskers','caps']:
    for it in bp[e]: it.set_linewidth(.6)
ax.axhline(1,color='k',ls='--',lw=.8)
ax.set_xticklabels([l for _,l in meth]); ax.set_ylabel(r'$P_t\,/\,P_{\mathrm{DSM}}$')
ax.set_ylim(0,3.2)
for i,(c,_) in enumerate(meth,1):
    r = (y/df[c].clip(lower=1e-6)).values
    ax.text(i,2.98,f'CoV\n{r.std()/r.mean():.3f}',ha='center',fontsize=6.5,
            linespacing=1.3,va='top')
plab(ax,'(b)')

ax = axes[2]
r = (y/df['PDSM_s(kN)'].clip(lower=1e-6)).values
lam = df['λc_b'].values
ax.scatter(lam,r,s=7,alpha=.45,color=BLUE,edgecolors='none')
ax.axhline(1,color='k',ls='--',lw=.8)
bins = np.array([0,.5,1.0,1.5,2.0,6.0]); ctr=[]; mu=[]
for a,b in zip(bins[:-1],bins[1:]):
    m=(lam>=a)&(lam<b)
    if m.sum()>5: ctr.append(min((a+b)/2,2.6)); mu.append(r[m].mean())
ax.plot(ctr,mu,'-o',color='#B00000',ms=3.5,lw=1.2,zorder=5,label='Bin mean')
ax.set_xlim(0,3); ax.set_ylim(0,3.2)
ax.set_xlabel(r'Global slenderness $\lambda_c$'); ax.set_ylabel(r'$P_t\,/\,P_{\mathrm{DSM},s}$')
ax.legend(loc='upper left')
n_hi = int((lam>3).sum())
ax.text(.97,.055,f'last bin: $\\lambda_c>2$ (n = {int(((lam>=2)).sum())})\n'
                 f'{n_hi} specimens with $\\lambda_c>3$ not shown',
        transform=ax.transAxes,ha='right',va='bottom',fontsize=6.3,
        color='#555555',linespacing=1.35)
plab(ax,'(c)')
fig.tight_layout(w_pad=2.2)
for e in ('png','pdf'): fig.savefig(OUT/f'Fig2-4_dsm_accuracy.{e}')
plt.close(fig); print('Fig2-4 ok')