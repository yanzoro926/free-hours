# ╭──────────────────────────────────────────────────────────────╮
# │  ＡＵＲＯＲＡＬ  ＣＯＤＥＸ  ·  极 光 手 稿                     │
# ╰──────────────────────────────────────────────────────────────╯
#
#           ~*~  a procedural poem that paints the sky  ~*~
#     the source is the art · the art is the source · run me
#
#   inspired by: IOCCC 2025 · endoh2 (lichtenberg curves)
#                IOCCC 2025 · tompng  (synthetic seashore)
#   seed: no two auroras are ever the same · entropy is sacred
#
# ── the noise · the breath · the rhythm ─────────────────────
from os import get_terminal_size as G
from time import time as T,sleep as Z
from math import floor as F,exp as E,sin as S
from random import seed as RD,random as R
from signal import signal as SG,SIGINT as I,SIGTERM as M
import sys as Y
# ── the loom · the warp · the weft ─────────────────────────
class N:  # noise field · chaos woven into order
 def __init__(s,e=None):e=e or int(T()*1e3)%2**31;RD(e);s.e=e;n=s.N=64;s.G=[[(R()*2-1,R()*2-1)for _ in range(n)]for _ in range(n)]
 def _F(s,t):return t*t*t*(t*(t*6-15)+10)                     # smoothstep
 def _L(s,a,b,t):return a+(b-a)*t                               # lerp
 def _D(s,x,y,a,b):g=s.G[b%s.N][a%s.N];return g[0]*(x-a)+g[1]*(y-b)
 def V(s,x,y,o=0):                                             # sample
  y+=o*.3;x0=F(x)%s.N;x1=(x0+1)%s.N;y0=F(y)%s.N;y1=(y0+1)%s.N
  sX=s._F(x-F(x));sY=s._F(y-F(y))
  return s._L(s._L(s._D(x,y,x0,y0),s._D(x,y,x1,y0),sX),s._L(s._D(x,y,x0,y1),s._D(x,y,x1,y1),sY),sY)
 def W(s,x,y,t=0,o=4,l=2.,g=.5):                              # fbm
  v=m=0.;a=1.;f=1.
  for _ in range(o):v+=a*s.V(x*f,y*f,t);m+=a;a*=g;f*=l
  return v/m
# ── the painter · the alchemist · the light-bringer ────────
class A:  # aurora renderer
 B=[(.25,.12,160,.8,1),(.30,.10,140,1.1,.9),(.20,.08,180,.6,.7),(.35,.15,120,.9,.85),(.40,.06,280,1.3,.6),(.15,.05,200,.5,.5)]
 def __init__(s,e=None):s.C=N(e);s.e=s.C.e;s.W=s.H=0;s.t=T();s.f=0;s.S=[];RD(s.e+42)
 def _H(s,h,u,l):                                            # hsl→rgb
  if u==0:v=int(l*255);return(v,v,v)
  def X(p,q,t):
   if t<0:t+=1
   if t>1:t-=1
   if t<1/6:return p+(q-p)*6*t
   if t<1/2:return q
   if t<2/3:return p+(q-p)*(2/3-t)*6
   return p
  q=l*(1+u)if l<.5 else l+u-l*u;p=2*l-q;n=(h%360)/360
  return(int(X(p,q,n+1/3)*255),int(X(p,q,n)*255),int(X(p,q,n-1/3)*255))
 def _K(s,r,c,t,b):                                         # aurora intensity
  n,x,u,p,o=b;h=r/max(s.H,1);y=c/(s.W*.3);v=t*u*.1
  d=s.C.W(y,v,t*.3,4);k=abs(h-n)/x
  return 0 if k>2 else max(0,min(1,(d*.7+.3)*E(-k*k)*o))
 def _P(s):                                                   # frame render
  s.W,s.H=G();t=T()-s.t;Q=[]
  for y in range(s.H):
   L=[]
   for x in range(s.W):
    Rr,Rg,Rb,z=0.,0.,0.,0.
    for b in s.B:
     i=s._K(y,x,t,b)
     if i>.01:r,g,b2=s._H(b[2],.8,i*.7);Rr+=r*i;Rg+=g*i;Rb+=b2*i;z+=i
    sr=0
    for sx,sy,p in s.S:
     if sx==x and sy==y:sr=(S(t*2+p)+1)/2*.6;break
    v=(z>.001);a=min(z,1)if v else 0
    rr=2+sr*255;gg=2+sr*255;bb=int(5+(1-y/max(s.H,1))*10)+sr*200
    if v:
     ar=Rr/z if z>0 else 0;ag=Rg/z if z>0 else 0;ab=Rb/z if z>0 else 0
     rr=int(rr*(1-a)+ar*a+sr*255);gg=int(gg*(1-a)+ag*a+sr*255);bb=int(bb*(1-a)+ab*a+sr*200)
    L.append(f"\x1b[48;2;{max(0,min(255,rr))};{max(0,min(255,gg))};{max(0,min(255,bb))}m ")
   L.append("\x1b[0m");Q.append("".join(L))
  return"".join(Q)
 def _Q(s,f=15,m=None):                                       # run loop
  Y.stdout.write("\x1b[?25l");Y.stdout.flush()
  def D(*a):Y.stdout.write("\x1b[?25h\x1b[0m\x1b[2J\x1b[H");Y.stdout.flush();Y.exit(0)
  SG(I,D);SG(M,D)
  try:
   d=1./f
   while 1:
    if m and s.f>=m:break
    o=T();f=s._P();Y.stdout.write("\x1b[H"+f);Y.stdout.flush();s.f+=1
    e=T()-o
    if e<d:Z(d-e)
  finally:D()
# ── the invocation · the breath before the first note ──────
if __name__=="__main__":
 Y.stdout.write("\x1b[2J\x1b[H");Y.stdout.flush()
 e=None
 if len(Y.argv)>1:
  try:e=int(Y.argv[1])
  except:e=hash(Y.argv[1])%2**31
 a=A(e)
 Y.stdout.write(f"\x1b[38;2;100;200;150m  ✦ Auroral Codex · 极光手稿 ✦  seed={a.e}\x1b[0m\n  Press Ctrl+C to exit\n");Y.stdout.flush();Z(1.5)
 a._Q(15)
# ── fin · the aurora fades but the code remains ───────────
