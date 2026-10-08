"""Sweep 6x36 danger-pooling policies on a frozen checkpoint."""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from perception_training.dataset import LidarPairDataset
from perception_training.model import SAUMNet


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--root",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--batch-size",type=int,default=256)
    args=ap.parse_args()
    ckpt=torch.load(args.checkpoint,map_location="cpu"); cfg=ckpt["config"]
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ns=cfg["sensor"]["network"]
    model=SAUMNet(height=ns["height"],width=ns["width"],vfov_deg=tuple(cfg["sensor"]["vfov_deg"]),
        base_channels=cfg["model"]["base_channels"],input_channels=4,
        reference_delta_t=cfg["model"]["reference_delta_t"],
        min_delta_t=cfg["model"]["min_delta_t"],max_delta_t=cfg["model"]["max_delta_t"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    ds=LidarPairDataset([args.root],default_delta_t=cfg["model"]["reference_delta_t"],
        min_delta_t=cfg["model"]["min_delta_t"],max_delta_t=cfg["model"]["max_delta_t"],cache_size=4)
    dl=DataLoader(ds,batch_size=args.batch_size,num_workers=4,pin_memory=True)
    thresholds=[0.30,0.40,0.50,0.60,0.70]
    policies=["speed_sigma","nearest","max_probability","closing","distance_closing"]
    accum={(t,p):dict(gt=0,det=0,sel=0,fp_cells=0,pred_cells=0) for t in thresholds for p in policies}
    with torch.inference_mode():
      for b in dl:
        b={k:v.to(device,non_blocking=True) for k,v in b.items()}
        out=model(b["input"],b["delta_t"])
        prob=torch.sigmoid(out["mask_logits"]); target=b["mask"]>0.5
        valid=b["input"][:,2:3]>0.5
        depth=b["input"][:,0:1].clamp_min(0.01)
        vel=out["velocity"]; sigma=torch.exp(out["log_sigma"]).mean(1,keepdim=True)
        speed=torch.linalg.vector_norm(vel,dim=1,keepdim=True)
        closing=(-vel[:,0:1]).clamp_min(0)
        scores={
          "speed_sigma":speed*sigma,
          "nearest":1.0/depth,
          "max_probability":prob,
          "closing":closing+1e-3/depth,
          "distance_closing":(closing+sigma)/(depth+0.10),
        }
        gt_cells=F.adaptive_max_pool2d((target&valid).float(),(6,36))>0.5
        gt=float(gt_cells.sum())
        for threshold in thresholds:
          pred=(prob>threshold)&valid
          pred_cells=F.adaptive_max_pool2d(pred.float(),(6,36))>0.5
          for policy,raw in scores.items():
            score=torch.where(pred,raw,torch.full_like(raw,-torch.inf))
            _,idx=F.adaptive_max_pool2d(score,(6,36),return_indices=True)
            selected_target=target.flatten(2).gather(2,idx.flatten(2)).view_as(gt_cells)
            selected_valid=valid.flatten(2).gather(2,idx.flatten(2)).view_as(gt_cells)
            a=accum[(threshold,policy)]
            a["gt"]+=gt
            a["det"]+=float((gt_cells&pred_cells).sum())
            a["sel"]+=float((gt_cells&pred_cells&selected_target&selected_valid).sum())
            a["fp_cells"]+=float((~gt_cells&pred_cells).sum())
            a["pred_cells"]+=float(pred_cells.sum())
    rows=[]
    for (threshold,policy),a in accum.items():
      rows.append({
        "threshold":threshold,"policy":policy,
        "cell_recall":a["det"]/a["gt"],
        "selected_target_retention":a["sel"]/a["gt"],
        "selection_precision_given_detected":a["sel"]/max(a["det"],1),
        "false_cell_fraction":a["fp_cells"]/max(a["pred_cells"],1),
        "gt_cell_support":a["gt"],
      })
    rows.sort(key=lambda x:x["selected_target_retention"],reverse=True)
    report={"checkpoint":args.checkpoint,"dataset":args.root,"output_size":[6,36],"results":rows}
    Path(args.output).write_text(yaml.safe_dump(report,sort_keys=False),encoding="utf-8")
    print(yaml.safe_dump(report,sort_keys=False),end="")


if __name__=="__main__":
    main()
