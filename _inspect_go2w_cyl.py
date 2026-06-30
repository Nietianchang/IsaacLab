from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
from pxr import Usd, UsdGeom, Gf
import numpy as np
top = "source/isaaclab_nav_task/isaaclab_nav_task/navigation/assets/data/Robots/go2w_cyl/go2w.usd"
stage = Usd.Stage.Open(top)
def av(t): return {"X":Gf.Vec3d(1,0,0),"Y":Gf.Vec3d(0,1,0),"Z":Gf.Vec3d(0,0,1)}[t]
L=[]
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    p=str(prim.GetPath())
    if "/FL_foot/collisions" in p or "/FR_foot/collisions" in p:
        t=prim.GetTypeName()
        if t=="Cylinder":
            c=UsdGeom.Cylinder(prim); ax=c.GetAxisAttr().Get(); r=c.GetRadiusAttr().Get(); h=c.GetHeightAttr().Get()
            m=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            w=Gf.Vec3d(av(ax)*m.ExtractRotationMatrix()).GetNormalized()
            L.append(f"CYL {p} axis={ax} r={r:.4f} h={h:.4f} world_axis=({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})")
        elif t=="Mesh":
            mm=UsdGeom.Mesh(prim); pts=mm.GetPointsAttr().Get()
            a=np.array([[v[0],v[1],v[2]] for v in pts]) if pts else np.zeros((1,3))
            ext=a.max(0)-a.min(0)
            L.append(f"MESH {p} npts={len(a)} ext=({ext[0]:.3f},{ext[1]:.3f},{ext[2]:.3f})")
        elif t not in ("Xform",):
            L.append(f"{t} {p}")
open("/tmp/cyl_result.txt","w").write(("\n".join(L) if L else "NOTHING UNDER collisions")+"\n")
simulation_app.close()
