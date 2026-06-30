from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
import carb
from pxr import Usd, UsdGeom, UsdPhysics
top = "source/isaaclab_nav_task/isaaclab_nav_task/navigation/assets/data/Robots/go2w_cyl/go2w.usd"
stage = Usd.Stage.Open(top)
L=[]
# global cylinder-approx setting
settings = carb.settings.get_settings()
for k in ["/physics/collisionApproximateCylinders","/physics/collisionConeCustomGeometry","/physics/collisionCylinderCustomGeometry"]:
    L.append(f"SETTING {k} = {settings.get(k)}")
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    p=str(prim.GetPath())
    if prim.GetTypeName()=="Cylinder" and "foot/collisions" in p:
        L.append(f"--- {p}")
        for a in prim.GetAttributes():
            n=a.GetName()
            if "physx" in n.lower() or "approx" in n.lower() or "collision" in n.lower():
                L.append(f"    {n} = {a.Get()}")
        L.append(f"    APIs: {[s for s in prim.GetAppliedSchemas()]}")
        break
open("/tmp/approx.txt","w").write("\n".join(L)+"\n")
simulation_app.close()
