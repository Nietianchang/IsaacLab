from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
from pxr import Usd, UsdGeom
import numpy as np
top = "source/isaaclab_nav_task/isaaclab_nav_task/navigation/assets/data/Robots/go2w/go2w.usd"
stage = Usd.Stage.Open(top)
L=[]
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    p=str(prim.GetPath())
    if "/FL_foot/collisions" in p or "/FR_foot/collisions" in p:
        t=prim.GetTypeName()
        if t=="Mesh":
            mm=UsdGeom.Mesh(prim); pts=mm.GetPointsAttr().Get()
            a=np.array([[v[0],v[1],v[2]] for v in pts]) if pts else np.zeros((1,3))
            mn=a.min(0);mx=a.max(0);ext=mx-mn
            L.append(f"MESH {p} npts={len(a)} min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f}) ext=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f})")
        elif t=="Cylinder":
            c=UsdGeom.Cylinder(prim);L.append(f"CYL {p} axis={c.GetAxisAttr().Get()} r={c.GetRadiusAttr().Get():.4f} h={c.GetHeightAttr().Get():.4f}")
        elif t not in ("Xform",):
            L.append(f"{t} {p}")
open("/tmp/orig_result.txt","w").write(("\n".join(L) if L else "NOTHING")+"\n")
simulation_app.close()
