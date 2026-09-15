import json, os, math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

MM = 1/25.4
WIDE = 174*MM  # Springer large-sized single-column width

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.1,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.dpi': 600,
})

# Exact manuscript values retained from verified result audits.
DATA = {
  'gpt2_l10h2': {
    'gain': [0.298451, 0.262940, 0.335184],
    'attn': [0.279460, 0.261643, 0.295510],
    'T': [0.782806, 0.721856, 0.838338],
    'R': [-0.341419, -0.371673, -0.310909],
    'O': [0.003864, 0.002363, 0.005329],
    'V': [-0.000349, -0.000810, 0.000104],
    'C': [-0.146451, -0.164685, -0.128324],
    'SCORE': [0.445252, 0.408802, 0.477107],
  },
  'gpt2_l10h10': {
    'gain': [-0.263442, -0.301914, -0.224875],
    'attn': [0.310250, 0.285276, 0.334539],
    'T': [0.328524, 0.300205, 0.355043],
    'R': [-0.125907, -0.138610, -0.113054],
    'O': [0.003634, 0.002586, 0.004619],
    'V': [0.001556, 0.001019, 0.002090],
    'C': [-0.471249, -0.514272, -0.427467],
    'SCORE': [0.206251, 0.185089, 0.225661],
  },
  'pythia_l8h2': {
    'gain': [0.485653, 0.326486, 0.721815],
    'T': [-0.143737, -0.247435, -0.084085],
    'R': [0.117800, 0.058225, 0.219125],
    'O': [-0.000270, -0.000847, 0.000256],
    'V': [-0.002099, -0.002762, -0.001557],
    'C': [0.513958, 0.354460, 0.752149],
  },
  'validation': {
    'T': 0.2689520684,
    'R': -0.4940997625,
    'O': 0.2134890065,
    'V': 0.1925102705,
    'C': 0.7112345942,
    'tc_total': 1.999999999996,
    'tc_T': 0.999999999998,
    'tc_C': 0.999999999998,
    'max_error': 1.11e-16,
  }
}
with open(os.path.join(os.path.dirname(__file__), 'figure_data.json'), 'w') as f:
    json.dump(DATA, f, indent=2)

FACTOR_COLORS = {'T':'C0','R':'C1','O':'C7','V':'C2','C':'C4'}
FACTOR_MARKERS = {'T':'o','R':'s','O':'D','V':'^','C':'P'}


def ci_err(t):
    m, lo, hi = t
    return np.array([[m-lo],[hi-m]])

def clean_axes(ax, zero_x=False, zero_y=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, width=0.8)
    if zero_x: ax.axvline(0, color='0.45', lw=0.8, zorder=0)
    if zero_y: ax.axhline(0, color='0.45', lw=0.8, zorder=0)


def panel_label(ax, s):
    ax.text(-0.08, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')


def save_all(fig, stem):
    for ext in ('pdf','svg','eps','png'):
        path = os.path.join(OUT, f'{stem}.{ext}')
        kwargs = {'bbox_inches':'tight'}
        if ext == 'png': kwargs['dpi'] = 600
        fig.savefig(path, **kwargs)
    plt.close(fig)

# ---------------- Figure 1: conceptual overview ----------------
fig = plt.figure(figsize=(WIDE, 3.65))
gs = GridSpec(2,2,figure=fig,height_ratios=[1,1.05],width_ratios=[1.08,0.92],hspace=0.35,wspace=0.28)
axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[1,:])
for ax in (axA,axB,axC): ax.set_axis_off()
panel_label(axA,'(a)'); panel_label(axB,'(b)'); panel_label(axC,'(c)')

def state_row(ax, y, label, utility, perturb=False):
    ax.text(0.02,y+0.14,label,transform=ax.transAxes,fontweight='bold',fontsize=9)
    box=FancyBboxPatch((0.03,y),0.14,0.11,boxstyle='round,pad=0.01',fc='white',ec='0.25',lw=1,transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(0.10,y+0.055,'primary',transform=ax.transAxes,color='0.35',ha='center',va='center',fontsize=8.3)
    hb=FancyBboxPatch((0.20,y+0.018),0.115,0.074,boxstyle='round,pad=0.01',fc='white',ec='C0',lw=1.2,transform=ax.transAxes)
    ax.add_patch(hb)
    ax.text(0.257,y+0.055,r'head $h$',transform=ax.transAxes,color='C0',ha='center',va='center',fontsize=8.5)
    if perturb:
        ax.plot([0.185,0.205],[y+0.025,y+0.095],color='C1',lw=1.8,transform=ax.transAxes,clip_on=False)
        ax.plot([0.205,0.185],[y+0.025,y+0.095],color='C1',lw=1.8,transform=ax.transAxes,clip_on=False)
        start=0.39
    else:
        start=0.39
    ax.add_patch(FancyArrowPatch((0.34,y+0.055),(0.41,y+0.055),arrowstyle='-|>',mutation_scale=10,lw=0.9,transform=ax.transAxes))
    ax.text(0.43,y+0.055,utility,transform=ax.transAxes,va='center',fontsize=8.5)

state_row(axA,0.56,'Intact state',r'$E_A(h)$')
state_row(axA,0.24,'Perturbed state',r'$E_B(h)$',True)
axB.text(0.02,0.76,r'$\Delta_h = E_B(h)-E_A(h)$',transform=axB.transAxes,fontsize=16)
axB.text(0.02,0.50,'Detection: did causal usefulness change?',transform=axB.transAxes,fontsize=9)
axB.text(0.02,0.30,'FALI: what accounts for that change?',transform=axB.transAxes,fontweight='bold',fontsize=9)

labels=[('T','target score(s)'),('R','rival scores'),('O','other valid scores'),('V','value content'),('C','surrounding context')]
xs=np.linspace(0.10,0.90,5)
for (lab,desc),x in zip(labels,xs):
    rect=FancyBboxPatch((x-0.065,0.57),0.13,0.18,boxstyle='round,pad=0.015',fc='white',ec=FACTOR_COLORS[lab],lw=1.2,transform=axC.transAxes)
    axC.add_patch(rect)
    axC.text(x,0.66,lab,transform=axC.transAxes,ha='center',va='center',fontweight='bold',fontsize=11,color=FACTOR_COLORS[lab])
    axC.text(x,0.46,desc,transform=axC.transAxes,ha='center',fontsize=8)
axC.text(0.50,0.20,r'$\phi_T+\phi_R+\phi_O+\phi_V+\phi_C=\Delta_h$',transform=axC.transAxes,ha='center',fontsize=11)
save_all(fig,'Fig1')

# ---------------- Figure 2: intervention game ----------------
fig = plt.figure(figsize=(WIDE, 4.15))
gs=GridSpec(2,2,figure=fig,height_ratios=[1,0.9],width_ratios=[1,1.08],hspace=0.30,wspace=0.30)
ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,:])
for ax in (ax1,ax2,ax3): ax.set_axis_off()
panel_label(ax1,'(a)'); panel_label(ax2,'(b)'); panel_label(ax3,'(c)')
# score matrix
x0,y0,w,h=0.13,0.18,0.50,0.62; n=6; cw=w/n; ch=h/n
for i in range(n):
    for j in range(n):
        masked=j>i
        ax1.add_patch(Rectangle((x0+j*cw,y0+(n-1-i)*ch),cw,ch,fc='0.92' if masked else 'white',ec='0.72',lw=0.7,hatch='///' if masked else None,transform=ax1.transAxes))
for j in range(5):
    lab='T' if j==2 else 'R'; col=FACTOR_COLORS[lab]
    ax1.add_patch(Rectangle((x0+j*cw,y0),cw,ch,fc='white',ec=col,lw=1.2,transform=ax1.transAxes))
    ax1.text(x0+(j+0.5)*cw,y0+0.5*ch,lab,transform=ax1.transAxes,ha='center',va='center',fontweight='bold',color=col,fontsize=9)
for i,j in [(0,0),(1,0),(1,1),(2,0),(2,1),(2,2),(3,0),(3,3),(4,1),(4,4)]:
    ax1.text(x0+(j+0.5)*cw,y0+(n-1-i+0.5)*ch,'O',transform=ax1.transAxes,ha='center',va='center',fontweight='bold',color=FACTOR_COLORS['O'],fontsize=9)
vx=x0+w+0.10
for i in range(n): ax1.add_patch(Rectangle((vx,y0+i*ch),0.045,ch,fc='0.90',ec='0.80',lw=0.6,transform=ax1.transAxes))
ax1.text(vx+0.022,y0+h+0.035,'V',transform=ax1.transAxes,ha='center',fontweight='bold',color=FACTOR_COLORS['V'],fontsize=10)
ax1.text(0.05,0.48,'query $q$',transform=ax1.transAxes,rotation=90,va='center',fontsize=9)
ax1.text(0.34,0.10,'key $k$',transform=ax1.transAxes,ha='center',fontsize=9)
ax1.text(0.56,0.13,'$q^*$',transform=ax1.transAxes,fontsize=9)
ax1.text(vx+0.025,0.49,'mask',transform=ax1.transAxes,ha='center',fontsize=8,color='0.4')

ax2.text(0.08,0.85,'T   R   O   V   C',transform=ax2.transAxes,fontfamily='monospace',fontsize=10)
steps=['Hybrid scores','Masked softmax','Pattern $P$','Candidate $P\odot V$','Candidate output $z$','Inject full $z$']
y=0.73
for s in steps:
    bb=FancyBboxPatch((0.13,y),0.66,0.075,boxstyle='round,pad=0.015',fc='white',ec='0.35',lw=0.8,transform=ax2.transAxes)
    ax2.add_patch(bb); ax2.text(0.46,y+0.0375,s,transform=ax2.transAxes,ha='center',va='center',fontsize=8.5); y-=0.092
ax2.text(0.06,0.05,r'$U(\mathrm{cell})=m(\mathrm{hybrid},h)-m(\mathrm{candidate\ zero},\mathrm{same}\ C)$',transform=ax2.transAxes,fontsize=8.2)

ax3.text(0.08,0.68,r'$2^5=32$ hybrid cells',transform=ax3.transAxes,fontweight='bold',fontsize=10)
ax3.text(0.36,0.72,r'$\mathrm{SCORE}=\{T,R,O\}$',transform=ax3.transAxes,fontsize=10)
ax3.text(0.36,0.51,r'$\mathrm{VALUE}=\{V\}$',transform=ax3.transAxes,fontsize=10)
ax3.text(0.36,0.30,r'$\mathrm{CONTEXT}=\{C\}$',transform=ax3.transAxes,fontsize=10)
ax3.text(0.68,0.65,r'$\Phi_{\mathrm{SCORE}}=\phi_T+\phi_R+\phi_O$',transform=ax3.transAxes,fontsize=10)
ax3.text(0.68,0.38,r'$\Phi_{\mathrm{SCORE}}+\phi_V+\phi_C=\Delta_h$',transform=ax3.transAxes,fontsize=10)
ax3.text(0.50,0.08,'Standard Owen value; 36 block-contiguous admissible permutations',transform=ax3.transAxes,ha='center',fontsize=8.5)
save_all(fig,'Fig2')

# ---------------- Figure 3: validation ----------------
fig=plt.figure(figsize=(WIDE,4.8))
gs=GridSpec(2,2,figure=fig,height_ratios=[1,0.9],width_ratios=[1.15,1],hspace=0.35,wspace=0.35)
ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,:])
panel_label(ax1,'(a)'); panel_label(ax2,'(b)'); panel_label(ax3,'(c)')
vals=np.eye(5)
ax1.imshow(vals,cmap='Greys',vmin=0,vmax=1,aspect='equal')
ax1.set_xticks(range(5),[r'$\phi_T$',r'$\phi_R$',r'$\phi_O$',r'$\phi_V$',r'$\phi_C$'])
ax1.set_yticks(range(5),['T only','R only','O only','V only','C only'])
for i in range(5):
    for j in range(5): ax1.text(j,i,'1.00' if i==j else '0',ha='center',va='center',fontsize=8,color='white' if i==j else 'black')
for i,(k,d) in enumerate(zip(['T','R','O','V','C'],[DATA['validation'][x] for x in ['T','R','O','V','C']])):
    ax1.text(5.25,i,f'{d:+.3f}',va='center',fontsize=8)
ax1.text(5.25,-0.80,r'$\Delta U$',fontweight='bold',fontsize=9)
ax1.set_xlim(-0.5,5.9)
for sp in ax1.spines.values(): sp.set_visible(False)

cats=[r'$\phi_T$',r'$\phi_C$',r'$\Delta U$']
barvals=[DATA['validation']['tc_T'],DATA['validation']['tc_C'],DATA['validation']['tc_total']]
ax2.barh(range(3),barvals,color=['C0','C4','0.2'],height=0.55)
ax2.set_yticks(range(3),cats)
ax2.set_xlabel('utility contribution')
ax2.set_xlim(0,2.2); clean_axes(ax2)
for i,v in enumerate(barvals): ax2.text(v+0.04,i,f'{v:.1f}',va='center',fontsize=8)

ax3.set_axis_off()
rows=[('Score to pattern reconstruction','0'),('Pattern/value to candidate output','0'),('Mask and batch-specific partitions','PASS'),('Independent endpoint recovery','0'),('Candidate isolation / live leakage','0'),('Coarse-fine quotient discrepancy',r'$1.11\times10^{-16}$')]
y=0.84
for name,status in rows:
    ax3.text(0.03,y,name,transform=ax3.transAxes,fontsize=8.8)
    ax3.text(0.80,y,status,transform=ax3.transAxes,fontsize=8.8,ha='right')
    y-=0.12
ax3.plot([0.03,0.82],[0.12,0.12],transform=ax3.transAxes,color='0.65',lw=0.8)
ax3.text(0.03,0.02,'Overall validation: PASS',transform=ax3.transAxes,fontweight='bold',fontsize=9.5)
save_all(fig,'Fig3')

# ---------------- Figure 4: GPT-2 IOI ----------------
A=DATA['gpt2_l10h2']; B=DATA['gpt2_l10h10']
fig=plt.figure(figsize=(WIDE,8.2))
gs=GridSpec(3,2,figure=fig,height_ratios=[0.78,1.15,1.20],hspace=0.38,wspace=0.22)
axA=fig.add_subplot(gs[0,:]); axB1=fig.add_subplot(gs[1,0]); axB2=fig.add_subplot(gs[1,1]); axC1=fig.add_subplot(gs[2,0]); axC2=fig.add_subplot(gs[2,1],sharey=axC1)
panel_label(axA,'(a)'); panel_label(axB1,'(b)'); panel_label(axC1,'(c)')
# panel a
clean_axes(axA,zero_y=True)
for label,marker,data in [('L10H2','o',A),('L10H10','s',B)]:
    x,y=data['attn'][0],data['gain'][0]
    axA.errorbar(x,y,xerr=ci_err(data['attn']),yerr=ci_err(data['gain']),fmt=marker,ms=6,capsize=3,lw=1.1,label=label,mfc='white',mew=1.2)
    axA.annotate(label,(x,y),xytext=(6,5),textcoords='offset points',fontsize=8.5,fontweight='bold')
axA.set_xlabel(r'$\Delta$ target attention'); axA.set_ylabel('conditional gain'); axA.set_xlim(0.25,0.35); axA.set_ylim(-0.36,0.38)
# panel b fine profiles
for ax,data,label in [(axB1,A,'L10H2'),(axB2,B,'L10H10')]:
    clean_axes(ax,zero_x=True)
    keys=['T','R','O','V','C']; ypos=np.arange(len(keys))
    for i,k in enumerate(keys):
        m,lo,hi=data[k]
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt=FACTOR_MARKERS[k],ms=5,capsize=3,mfc='white',mec=FACTOR_COLORS[k],ecolor=FACTOR_COLORS[k],mew=1.2,lw=1.0)
        txt=f'{m:+.3f}' if abs(m)>=0.01 else f'{m:+.4f}'
        ax.text(m+(0.035 if m>=0 else -0.035),i,txt,ha='left' if m>=0 else 'right',va='center',fontsize=7.7)
    ax.set_yticks(ypos,[r'$\phi_T$',r'$\phi_R$',r'$\phi_O$',r'$\phi_V$',r'$\phi_C$'])
    ax.invert_yaxis(); ax.set_xlabel('Owen attribution (95% CI)'); ax.set_xlim(-0.60,0.95)
    ax.text(0.50,1.02,label,transform=ax.transAxes,ha='center',fontweight='bold',fontsize=8.8)
axB2.set_yticklabels([])
# panel c waterfall
for ax,data,label in [(axC1,A,'L10H2'),(axC2,B,'L10H10')]:
    clean_axes(ax,zero_y=True)
    score,value,context,gain=data['SCORE'][0],data['V'][0],data['C'][0],data['gain'][0]
    level=0
    for idx,(name,val,hatch) in enumerate([('SCORE',score,'///'),('VALUE',value,'..'),('CONTEXT',context,'xx')]):
        new=level+val; bottom=min(level,new)
        ax.bar(idx,abs(val),bottom=bottom,width=0.58,facecolor='white',edgecolor='0.2',hatch=hatch,lw=0.9)
        if idx<2: ax.plot([idx+0.29,idx+1-0.29],[new,new],ls='--',color='0.35',lw=0.8)
        ax.text(idx,new+(0.035 if val>=0 else -0.045),f'{val:+.3f}' if abs(val)>=0.01 else f'{val:+.4f}',ha='center',va='bottom' if val>=0 else 'top',fontsize=7.7)
        level=new
    ax.bar(3,abs(gain),bottom=min(0,gain),width=0.58,facecolor='0.80',edgecolor='0.1',lw=0.9)
    ax.text(3,gain+(0.035 if gain>=0 else -0.045),f'{gain:+.3f}',ha='center',va='bottom' if gain>=0 else 'top',fontsize=8,fontweight='bold')
    ax.set_xticks(range(4),['SCORE','VALUE','CONTEXT','GAIN']); ax.set_ylim(-0.60,0.55)
    ax.text(0.50,1.02,label,transform=ax.transAxes,ha='center',fontweight='bold',fontsize=8.8)
axC1.set_ylabel('contribution to conditional gain'); axC2.set_yticklabels([])
save_all(fig,'Fig4')

# ---------------- Figure 5: cross-setting comparison ----------------
P=DATA['pythia_l8h2']
fig=plt.figure(figsize=(WIDE,5.4))
gs=GridSpec(2,2,figure=fig,height_ratios=[0.85,1.15],hspace=0.38,wspace=0.28)
ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,0]); ax4=fig.add_subplot(gs[1,1])
panel_label(ax1,'(a)'); panel_label(ax3,'(b)')
# top: gains on task-specific scales
for ax,data,label,ylabel in [(ax1,A,'GPT-2 IOI - L10H2','conditional gain\n(IO-S logit units)'),(ax2,P,'Pythia induction - L8H2','conditional gain\n(target log-prob units)')]:
    clean_axes(ax,zero_y=True)
    m,lo,hi=data['gain']; ax.errorbar([0],[m],yerr=[[m-lo],[hi-m]],fmt='D',ms=6,capsize=3,mfc='white',mew=1.2)
    ax.set_xlim(-0.4,0.4); ax.set_xticks([]); ax.set_ylabel(ylabel); ax.text(0.5,1.02,label,transform=ax.transAxes,ha='center',fontweight='bold',fontsize=8.8)
    ax.text(0.05,m,f'{m:+.3f}',va='center',fontsize=8)
ax1.set_ylim(-0.10,0.42); ax2.set_ylim(-0.10,0.82)
# bottom: T/R/C
for ax,data,label,xlim in [(ax3,A,'GPT-2 IOI',(-0.55,0.92)),(ax4,P,'Pythia induction',(-0.30,0.82))]:
    clean_axes(ax,zero_x=True)
    keys=['T','R','C']; ypos=np.arange(3)
    for i,k in enumerate(keys):
        m,lo,hi=data[k]
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt=FACTOR_MARKERS[k],ms=5,capsize=3,mfc='white',mec=FACTOR_COLORS[k],ecolor=FACTOR_COLORS[k],mew=1.2,lw=1.0)
        ax.text(m+(0.03 if m>=0 else -0.03),i,f'{m:+.3f}',ha='left' if m>=0 else 'right',va='center',fontsize=7.7)
    ax.set_yticks(ypos,[r'$\phi_T$',r'$\phi_R$',r'$\phi_C$']); ax.invert_yaxis(); ax.set_xlabel('FALI attribution'); ax.set_xlim(*xlim); ax.text(0.5,1.02,label,transform=ax.transAxes,ha='center',fontweight='bold',fontsize=8.8)
ax4.tick_params(labelleft=False)
save_all(fig,'Fig5')

print('Wrote figures to', OUT)
