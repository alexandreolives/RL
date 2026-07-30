"""Screen all hybrid schedules for finite outputs and gradient stability."""
from __future__ import annotations
import argparse, json, torch
from scripts.benchmark_hybrid_schedules import SCHEDULES
from models.atoms.hybrid import ConfigurableHybridCore


def main():
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cpu"); p.add_argument("--d-model", type=int, default=64); p.add_argument("--sequence", type=int, default=32); p.add_argument("--batch", type=int, default=2); p.add_argument("--iterations", type=int, nargs="+", default=[1,2,3,4]); p.add_argument("--carry-kda-state", action="store_true")
    a = p.parse_args(); device=torch.device(a.device); torch.manual_seed(0)
    rows=[]
    for name, stages in SCHEDULES.items():
        for loops in a.iterations:
            model=ConfigurableHybridCore(a.d_model, stages=stages, carry_kda_state=a.carry_kda_state).to(device).train()
            x=torch.randn(a.batch,a.sequence,a.d_model,device=device,requires_grad=True)
            y=model(x,iterations=loops); loss=y.float().square().mean(); loss.backward()
            grads=[p.grad.detach().float().norm().item() for p in model.parameters() if p.grad is not None]
            rows.append({"schedule":name,"iterations":loops,"carry_kda_state":a.carry_kda_state,"finite":bool(torch.isfinite(y).all() and torch.isfinite(loss)),"output_rms":float(y.detach().float().pow(2).mean().sqrt()),"input_grad_norm":float(x.grad.detach().float().norm()),"max_param_grad":max(grads,default=0.0)})
    print(json.dumps({"device":str(device),"results":rows},indent=2))


if __name__ == "__main__": main()
