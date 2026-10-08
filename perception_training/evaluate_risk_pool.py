"""Evaluate whether pooling preserves oracle CBF/TTC risk on static-sensor samples."""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from perception_training.dataset import LidarPairDataset
from perception_training.model import SAUMNet
from perception_training.geometry import absolute_to_relative_spherical_velocity


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--root",required=True,nargs="+"); ap.add_argument("--output",required=True)
    ap.add_argument("--batch-size",type=int,default=256); ap.add_argument("--mask-threshold",type=float,default=0.5)
    args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location="cpu"); cfg=ck["config"]; dev=torch.device("cuda:0")
    ns=cfg["sensor"]["network"]
    m=SAUMNet(height=ns["height"],width=ns["width"],vfov_deg=tuple(cfg["sensor"]["vfov_deg"]),
      base_channels=16,input_channels=4,reference_delta_t=cfg["model"]["reference_delta_t"],
      min_delta_t=cfg["model"]["min_delta_t"],max_delta_t=cfg["model"]["max_delta_t"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    ds=LidarPairDataset(args.root,default_delta_t=cfg["model"]["reference_delta_t"],
      min_delta_t=cfg["model"]["min_delta_t"],max_delta_t=cfg["model"]["max_delta_t"],cache_size=4)
    dl=DataLoader(ds,batch_size=args.batch_size,num_workers=4,pin_memory=True)
    policies=["speed_sigma","nearest","max_probability","closing","pred_cbf","pred_ttc"]
    stats={p:{"risk_cells":0,"selected_detected":0,"selected_positive_risk":0,"risk_ratio_sum":0.0,"oracle_exact":0} for p in policies}
    sample_count=dynamic_sample_count=0
    full_relative_samples=legacy_static_samples=0
    safe=0.5; gain=1.2; horizon=1.2; threshold=args.mask_threshold
    with torch.inference_mode():
      for b in dl:
        dynamic_sample=(b["mask"].flatten(1).sum(1)>0)
        has_ego_vector="ego_velocity_body_mps" in b
        if has_ego_vector:
          keep=dynamic_sample
        else:
          static=(b["ego_linear_speed_mps"].reshape(-1)<1e-5)
          keep=static & dynamic_sample
        if not keep.any(): continue
        b={k:v[keep].to(dev,non_blocking=True) for k,v in b.items()}
        sample_count+=len(b["input"]); dynamic_sample_count+=int((b["mask"].flatten(1).sum(1)>0).sum())
        if has_ego_vector: full_relative_samples+=len(b["input"])
        else: legacy_static_samples+=len(b["input"])
        o=m(b["input"],b["delta_t"]); prob=torch.sigmoid(o["mask_logits"])
        valid=b["input"][:,2:3]>0.5; pred=(prob>threshold)&valid; gt=(b["mask"]>0.5)&valid
        depth=b["input"][:,0:1]*10.0
        if has_ego_vector:
          gt_velocity=absolute_to_relative_spherical_velocity(
            b["velocity"],b["ego_velocity_body_mps"],tuple(cfg["sensor"]["vfov_deg"]))
        else:
          # Legacy shards can only be evaluated where the sensor is static.
          gt_velocity=b["velocity"]
        gt_vr=gt_velocity[:,0:1]
        gt_approach=(-gt_vr).clamp_min(0)
        gt_cbf=torch.relu(-(gt_vr+gain*(depth-safe)))
        gt_ttc=depth/(gt_approach+1e-3)
        oracle_risk=gt_cbf*(0.55+0.45*torch.exp(-gt_ttc/horizon))*(0.55+0.45*torch.exp(-depth/safe))
        oracle_risk=torch.where(gt,oracle_risk,torch.zeros_like(oracle_risk))
        oracle_max,oracle_idx=F.adaptive_max_pool2d(oracle_risk,(6,36),return_indices=True)
        risk_cells=oracle_max>0
        vel=o["velocity"]; sigma=torch.exp(o["log_sigma"]).mean(1,keepdim=True)
        if has_ego_vector:
          relative_vel=absolute_to_relative_spherical_velocity(
            vel,b["ego_velocity_body_mps"],tuple(cfg["sensor"]["vfov_deg"]))
        else:
          relative_vel=vel
        pred_vr=relative_vel[:,0:1]; pred_approach=(-pred_vr).clamp_min(0)
        pred_cbf=torch.relu(-((pred_vr-2*sigma)+gain*(depth-safe)))
        pred_ttc=depth/(torch.relu(-(pred_vr-2*sigma))+1e-3)
        scores={
          "speed_sigma":torch.linalg.vector_norm(relative_vel,dim=1,keepdim=True)*sigma,
          "nearest":1/(depth+1e-3),
          "max_probability":prob,
          "closing":pred_approach,
          "pred_cbf":pred_cbf,
          "pred_ttc":torch.exp(-pred_ttc/horizon)/(depth+0.1),
        }
        for name,raw in scores.items():
          score=torch.where(pred,raw,torch.full_like(raw,-torch.inf))
          _,idx=F.adaptive_max_pool2d(score,(6,36),return_indices=True)
          selected_pred=pred.flatten(2).gather(2,idx.flatten(2)).view_as(risk_cells)
          selected_risk=oracle_risk.flatten(2).gather(2,idx.flatten(2)).view_as(risk_cells)
          s=stats[name]; n=float(risk_cells.sum())
          s["risk_cells"]+=n
          s["selected_detected"]+=float((risk_cells&selected_pred).sum())
          s["selected_positive_risk"]+=float((risk_cells&(selected_risk>0)).sum())
          s["risk_ratio_sum"]+=float(torch.where(risk_cells,selected_risk/oracle_max.clamp_min(1e-9),torch.zeros_like(selected_risk)).sum())
          s["oracle_exact"]+=float((risk_cells&(idx==oracle_idx)).sum())
    results=[]
    for name,s in stats.items():
      n=max(s["risk_cells"],1)
      results.append({"policy":name,"risk_cell_support":s["risk_cells"],
        "prediction_available":s["selected_detected"]/n,
        "positive_risk_target_retention":s["selected_positive_risk"]/n,
        "mean_oracle_risk_fraction":s["risk_ratio_sum"]/n,
        "exact_oracle_winner_rate":s["oracle_exact"]/n})
    results.sort(key=lambda x:x["mean_oracle_risk_fraction"],reverse=True)
    if legacy_static_samples==0:
      scope="all dynamic samples using body-frame 3D ego velocity"
    elif full_relative_samples==0:
      scope="legacy dynamic samples with static sensor only"
    else:
      scope="mixed: full relative velocity where available, static-only legacy fallback"
    report={"scope":scope,"evaluated_samples":sample_count,
      "full_relative_samples":full_relative_samples,"legacy_static_samples":legacy_static_samples,
      "oracle":"ground-truth CBF violation x TTC weight x distance weight","safe_distance_m":safe,
      "barrier_gain":gain,"ttc_horizon_s":horizon,"mask_threshold":threshold,"results":results}
    Path(args.output).write_text(yaml.safe_dump(report,sort_keys=False),encoding="utf-8")
    print(yaml.safe_dump(report,sort_keys=False),end="")


if __name__=="__main__": main()
